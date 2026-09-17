"""Trading 212 Public API (v0) client.

A thin, dependency-light wrapper around the Trading 212 Public API. Reference:
https://docs.trading212.com/api

Authentication
--------------
Every request uses HTTP Basic auth: the API Key as the username and the API
Secret as the password (Base64-encoded ``API_KEY:API_SECRET``). Older keys that
authenticate with a single value via the ``Authorization`` header are also
supported by passing only ``api_key`` (leave ``api_secret`` empty).

Environments
------------
* Paper / demo : https://demo.trading212.com/api/v0
* Live / real  : https://live.trading212.com/api/v0

Only Invest and Stocks ISA account types are supported by the API. Orders can
be placed only in the account's primary currency, and a *negative* ``quantity``
denotes a sell order.

Configuration (environment variables / .env)
--------------------------------------------
    T212_API_KEY      – API key (required)
    T212_API_SECRET   – API secret (required for Basic auth; omit for legacy keys)
    T212_ENV          – "demo" (default) or "live"

"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Dict, Iterator, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

__all__ = ["Trading212Client", "Trading212Error", "TimeValidity"]

BASE_URLS: Dict[str, str] = {
    "demo": "https://demo.trading212.com/api/v0",
    "live": "https://live.trading212.com/api/v0",
}

# Allowed values for the ``timeValidity`` order field.
TimeValidity = ("DAY", "GOOD_TILL_CANCEL")


class Trading212Error(requests.HTTPError):
    """Raised when the Trading 212 API returns a non-2xx response.

    Exposes the parsed JSON error body (when available) on ``.payload``.
    """

    def __init__(self, message: str, response: Optional[requests.Response] = None):
        super().__init__(message, response=response)
        self.payload: Any = None
        if response is not None:
            try:
                self.payload = response.json()
            except ValueError:
                self.payload = response.text


class Trading212Client:
    """Client for the Trading 212 Public API (v0)."""

    def __init__(
        self,
        api_key: str,
        api_secret: str = "",
        environment: str = "demo",
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if environment not in BASE_URLS:
            raise ValueError(
                f"environment must be one of {sorted(BASE_URLS)}, got {environment!r}"
            )

        self.base_url = BASE_URLS[environment]
        self.environment = environment
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Authorization": self._build_auth_header(api_key, api_secret),
                "Accept": "application/json",
            }
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "Trading212Client":
        """Build a client from ``T212_*`` environment variables."""
        api_key = os.getenv("T212_API_KEY", "")
        if not api_key:
            raise ValueError("T212_API_KEY environment variable is not set")
        return cls(
            api_key=api_key,
            api_secret=os.getenv("T212_API_SECRET", ""),
            environment=os.getenv("T212_ENV", "live").lower(),
            **kwargs,
        )

    @staticmethod
    def _build_auth_header(api_key: str, api_secret: str) -> str:
        # HTTP Basic auth: base64("API_KEY:API_SECRET"), matching the documented
        # `echo -n "$KEY:$SECRET" | base64` credential.
        token = base64.b64encode(f"{api_key}:{api_secret}".encode("utf-8")).decode("ascii")
        return f"Basic {token}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Perform a request, transparently retrying on rate limits (429)."""
        # ``path`` may be an absolute API path (e.g. a nextPagePath) or a
        # relative endpoint appended to the base URL.
        if path.startswith("/api/"):
            url = self.base_url.split("/api/", 1)[0] + path
        else:
            url = f"{self.base_url}/{path.lstrip('/')}"

        attempt = 0
        while True:
            attempt += 1
            response = self._session.request(
                method,
                url,
                params=params,
                json=json,
                timeout=self.timeout,
            )

            if response.status_code == 429 and attempt <= self.max_retries:
                time.sleep(self._retry_after_seconds(response))
                continue

            if not response.ok:
                raise Trading212Error(
                    f"{method} {url} -> {response.status_code}: {response.text}",
                    response=response,
                )

            if not response.content:
                return None
            try:
                return response.json()
            except ValueError:
                return response.text

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> float:
        """Compute how long to wait before retrying a rate-limited request."""
        reset = response.headers.get("x-ratelimit-reset")
        if reset:
            try:
                wait = float(reset) - time.time()
                if wait > 0:
                    return min(wait, 60.0)
            except ValueError:
                pass
        return 1.0

    def _paginate(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Iterator[Dict[str, Any]]:
        """Yield every item across all pages of a cursor-paginated endpoint."""
        page = self._request("GET", path, params=params)
        while page:
            for item in page.get("items", []):
                yield item
            next_path = page.get("nextPagePath")
            if not next_path:
                break
            page = self._request("GET", next_path)

    def account_summary(self) -> Dict[str, Any]:
        """Cash and investment metrics for the account."""
        return self._request("GET", "equity/account/summary")

    def exchanges(self) -> Any:
        """All accessible exchanges and their working schedules."""
        return self._request("GET", "equity/metadata/exchanges")

    def instruments(self) -> Any:
        """All tradable instruments (tickers, ISINs, names, ...)."""
        return self._request("GET", "equity/metadata/instruments")

    def positions(self) -> Any:
        """All open positions for the account."""
        return self._request("GET", "equity/positions")

    def pending_orders(self) -> Any:
        """All currently active (unfilled) orders."""
        return self._request("GET", "equity/orders")

    def get_order(self, order_id: int) -> Dict[str, Any]:
        """Fetch a single pending order by its numeric id."""
        return self._request("GET", f"equity/orders/{order_id}")

    def cancel_order(self, order_id: int) -> Any:
        """Request cancellation of an active, unfilled order."""
        return self._request("DELETE", f"equity/orders/{order_id}")

    def place_market_order(
        self, ticker: str, quantity: float, *, extended_hours: bool = False
    ) -> Dict[str, Any]:
        """Place a market order. Negative ``quantity`` sells."""
        body = {
            "ticker": ticker,
            "quantity": quantity,
            "extendedHours": extended_hours,
        }
        return self._request("POST", "equity/orders/market", json=body)

    def place_limit_order(
        self,
        ticker: str,
        quantity: float,
        limit_price: float,
        *,
        time_validity: str = "DAY",
    ) -> Dict[str, Any]:
        """Place a limit order. Negative ``quantity`` sells."""
        self._check_time_validity(time_validity)
        body = {
            "ticker": ticker,
            "quantity": quantity,
            "limitPrice": limit_price,
            "timeValidity": time_validity,
        }
        return self._request("POST", "equity/orders/limit", json=body)

    def place_stop_order(
        self,
        ticker: str,
        quantity: float,
        stop_price: float,
        *,
        time_validity: str = "DAY",
    ) -> Dict[str, Any]:
        """Place a stop (stop-loss) order. Negative ``quantity`` sells."""
        self._check_time_validity(time_validity)
        body = {
            "ticker": ticker,
            "quantity": quantity,
            "stopPrice": stop_price,
            "timeValidity": time_validity,
        }
        return self._request("POST", "equity/orders/stop", json=body)

    def place_stop_limit_order(
        self,
        ticker: str,
        quantity: float,
        stop_price: float,
        limit_price: float,
        *,
        time_validity: str = "DAY",
    ) -> Dict[str, Any]:
        """Place a stop-limit order. Negative ``quantity`` sells."""
        self._check_time_validity(time_validity)
        body = {
            "ticker": ticker,
            "quantity": quantity,
            "stopPrice": stop_price,
            "limitPrice": limit_price,
            "timeValidity": time_validity,
        }
        return self._request("POST", "equity/orders/stop_limit", json=body)

    @staticmethod
    def _check_time_validity(value: str) -> None:
        if value not in TimeValidity:
            raise ValueError(
                f"time_validity must be one of {TimeValidity}, got {value!r}"
            )

    def history_orders(
        self, *, limit: int = 20, cursor: Optional[str] = None, ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        """One page of historical orders (see :meth:`iter_history_orders`)."""
        params = _page_params(limit=limit, cursor=cursor, ticker=ticker)
        return self._request("GET", "equity/history/orders", params=params)

    def iter_history_orders(
        self, *, limit: int = 50, ticker: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Iterate over every historical order across all pages."""
        return self._paginate(
            "equity/history/orders", _page_params(limit=limit, ticker=ticker)
        )

    def history_dividends(
        self, *, limit: int = 20, cursor: Optional[str] = None, ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        """One page of dividend payments."""
        params = _page_params(limit=limit, cursor=cursor, ticker=ticker)
        return self._request("GET", "equity/history/dividends", params=params)

    def iter_dividends(
        self, *, limit: int = 50, ticker: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Iterate over every dividend payment across all pages."""
        return self._paginate(
            "equity/history/dividends", _page_params(limit=limit, ticker=ticker)
        )

    def history_transactions(
        self, *, limit: int = 20, cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """One page of cash transactions."""
        params = _page_params(limit=limit, cursor=cursor)
        return self._request("GET", "equity/history/transactions", params=params)

    def iter_transactions(self, *, limit: int = 50) -> Iterator[Dict[str, Any]]:
        """Iterate over every cash transaction across all pages."""
        return self._paginate("equity/history/transactions", _page_params(limit=limit))

    def exports(self) -> Any:
        """List previously requested CSV exports."""
        return self._request("GET", "equity/history/exports")

    def request_export(
        self,
        time_from: str,
        time_to: str,
        *,
        include_dividends: bool = True,
        include_interest: bool = True,
        include_orders: bool = True,
        include_transactions: bool = True,
    ) -> Dict[str, Any]:
        """Request a CSV export for the given ISO 8601 time range."""
        body = {
            "dataIncluded": {
                "includeDividends": include_dividends,
                "includeInterest": include_interest,
                "includeOrders": include_orders,
                "includeTransactions": include_transactions,
            },
            "timeFrom": time_from,
            "timeTo": time_to,
        }
        return self._request("POST", "equity/history/exports", json=body)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "Trading212Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _page_params(**kwargs: Any) -> Dict[str, Any]:
    """Drop ``None`` values so they are not sent as query parameters."""
    return {k: v for k, v in kwargs.items() if v is not None}


if __name__ == "__main__":
    with Trading212Client.from_env() as t212:
        # print(json.dumps(t212.account_summary()))
        print(json.dumps(t212.positions()))
        # print(json.dumps(list(t212.iter_history_orders())))
        # print(json.dumps(list(t212.iter_transactions())))

