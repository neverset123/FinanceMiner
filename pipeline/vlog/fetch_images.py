"""
Fetch a relevant, freely-licensed photo per scene from Wikimedia Commons.

No API key required. Returns a local file path (downloaded) or None.
The caller composites the caption text over the image.

Chinese scripts are mapped to English search queries via a keyword table,
with an optional per-line override in the script using the syntax:

    某段中文旁白  @@  yen banknote, tokyo stock exchange

Everything after `@@` on a script line is treated as comma-separated English
search terms (best for Commons, which is mostly English-indexed).

Keyword mappings can be customized by creating a keywords.json file in the
pipeline/vlog directory with the format:
    {
        "套息": "japanese yen banknote",
        "美元": "us dollar banknote"
    }
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import requests

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "context-video-generator/1.0 (educational demo)"}


def _build_session() -> requests.Session:
    """Build a requests session respecting proxy and SSL settings from environment."""
    session = requests.Session()
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or None
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    verify = os.getenv("NODE_TLS_REJECT_UNAUTHORIZED", "1") != "0"
    session.verify = verify
    session.headers.update(HEADERS)
    return session


_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = _build_session()
    return _session
CACHE = Path("assets/images")
CACHE.mkdir(parents=True, exist_ok=True)
KEYWORDS_FILE = Path(__file__).parent / "keywords.json"

# Default keyword map - empty by default to encourage using keywords.json
# Finance defaults are in keywords.json.example - copy and customize it!
DEFAULT_KEYWORD_MAP: dict[str, str] = {}

DEFAULT_QUERY = "global finance economy"


def load_keyword_map() -> dict[str, str]:
    """Load keyword mappings from keywords.json if it exists, otherwise use defaults."""
    if KEYWORDS_FILE.exists():
        try:
            with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
                custom = json.load(f)
                print(f"[i] Loaded {len(custom)} custom keywords from {KEYWORDS_FILE}")
                return custom
        except Exception as e:
            print(f"[!] Failed to load {KEYWORDS_FILE}: {e}, using defaults")
    return DEFAULT_KEYWORD_MAP


def parse_line(raw: str) -> tuple[str, str | None]:
    """Split a script line into (spoken_text, explicit_query_or_None)."""
    if "@@" in raw:
        text, q = raw.split("@@", 1)
        return text.strip(), q.strip()
    return raw.strip(), None


def query_for(text: str, explicit: str | None, keyword_map: dict[str, str] | None = None) -> str:
    """
    Determine search query for the given text.
    
    Priority:
    1. Explicit query (from @@ syntax)
    2. Longest matching keyword in text
    3. Use text itself (first 50 chars, cleaned)
    4. Default query
    """
    if explicit:
        return explicit
    
    if keyword_map is None:
        keyword_map = load_keyword_map()
    
    # Sort keywords by length (longest first) to prefer more specific matches
    # e.g., "儲備貨幣" should match before "貨幣" if both are in the map
    sorted_keywords = sorted(keyword_map.items(), key=lambda x: len(x[0]), reverse=True)
    
    for zh, en in sorted_keywords:
        if zh in text:
            return en
    
    # Fallback: use the text itself (cleaned, first 50 chars)
    # This allows the system to work even without keyword mappings
    clean_text = re.sub(r'[^\w\s]', ' ', text).strip()
    if clean_text and len(clean_text) > 2:
        # Take meaningful portion, remove common stopwords
        clean_text = re.sub(r'^(第[一二三四五六七八九十]+|哈佬|大家好|歡迎|本期)', '', clean_text).strip()
        if clean_text:
            return clean_text[:50]
    
    return DEFAULT_QUERY


def _commons_search(query: str, limit: int = 8) -> list[str]:
    params = {
        "action": "query", "format": "json", "list": "search",
        "srsearch": f"{query} filetype:bitmap",
        "srnamespace": "6", "srlimit": str(limit),
    }
    session = _get_session()
    r = session.get(COMMONS_API, params=params, timeout=30)
    if r.status_code != 200:
        print(f"    [!] Commons search returned {r.status_code} for '{query}'")
        return []
    return [item["title"] for item in r.json().get("query", {}).get("search", [])]


def _thumb_url(title: str, width: int = 1280) -> str | None:
    params = {
        "action": "query", "format": "json", "prop": "imageinfo",
        "iiprop": "url|mime", "iiurlwidth": str(width), "titles": title,
    }
    session = _get_session()
    r = session.get(COMMONS_API, params=params, timeout=30)
    if r.status_code != 200:
        return None
    pages = r.json().get("query", {}).get("pages", {})
    for p in pages.values():
        info = p.get("imageinfo", [{}])[0]
        mime = info.get("mime", "")
        if mime.startswith("image/") and mime not in ("image/svg+xml",):
            return info.get("thumburl") or info.get("url")
    return None


def fetch_image(text: str, explicit: str | None = None, width: int = 1280, 
                keyword_map: dict[str, str] | None = None) -> tuple[str | None, str]:
    """Return (local_path_or_None, query_used)."""
    query = query_for(text, explicit, keyword_map)
    key = hashlib.md5(query.encode("utf-8")).hexdigest()[:12]
    for ext in (".jpg", ".jpeg", ".png"):
        cached = CACHE / f"{key}{ext}"
        if cached.exists():
            return str(cached), query
    try:
        session = _get_session()
        for title in _commons_search(query):
            url = _thumb_url(title, width)
            if not url:
                continue
            resp = session.get(url, timeout=40)
            if resp.status_code != 200 or not resp.content:
                continue
            ext = ".png" if url.lower().split("?")[0].endswith(".png") else ".jpg"
            out = CACHE / f"{key}{ext}"
            out.write_bytes(resp.content)
            return str(out), query
    except Exception as e:  # network blocked / proxy / timeout
        print(f"    [!] image fetch failed for '{query}': {type(e).__name__}")
    return None, query


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "日圓套息交易"
    print(fetch_image(t))
