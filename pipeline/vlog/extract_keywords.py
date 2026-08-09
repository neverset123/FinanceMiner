"""
LLM-based keyword extraction — self-contained, no agent/ imports.

Given a list of Chinese script lines, returns English image-search queries
(one per line) by asking the configured LLM to produce contextually relevant terms.

Uses environment variables for provider resolution (supports Anthropic, OpenAI,
Gemini, Bedrock, custom endpoints).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AuthProvider = Literal["anthropic", "openai", "gemini", "bedrock", "custom"]

PROVIDER_BASE_URLS: dict[AuthProvider, str] = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "bedrock": "",
    "custom": "",
}

PROVIDER_DEFAULT_MODELS: dict[AuthProvider, str] = {
    "anthropic": "anthropic::claude-4-5-sonnet",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "bedrock": "anthropic.claude-sonnet-4-20250514-v1:0",
    "custom": "default",
}


@dataclass
class LLMConfig:
    provider: AuthProvider = "custom"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Resolve LLM config from environment variables.

        Precedence:
        1. ANTHROPIC_API_KEY → Anthropic
        2. OPENAI_API_KEY → OpenAI
        3. GEMINI_API_KEY → Gemini
        4. CLAUDE_CODE_USE_BEDROCK → Bedrock
        5. BASE_URL + API_KEY → Custom
        """
        # 1. Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
        if anthropic_key:
            base_url = os.getenv("ANTHROPIC_BASE_URL", PROVIDER_BASE_URLS["anthropic"])
            if base_url and not base_url.endswith("/v1") and not base_url.endswith("/v1/"):
                base_url = base_url.rstrip("/") + "/v1"
            return cls(
                provider="anthropic",
                api_key=anthropic_key,
                base_url=base_url,
                model=os.getenv("CHAT_MODEL", PROVIDER_DEFAULT_MODELS["anthropic"]),
            )

        # 2. OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            return cls(
                provider="openai",
                api_key=openai_key,
                base_url=os.getenv("BASE_URL", PROVIDER_BASE_URLS["openai"]),
                model=os.getenv("CHAT_MODEL", PROVIDER_DEFAULT_MODELS["openai"]),
            )

        # 3. Gemini
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            return cls(
                provider="gemini",
                api_key=gemini_key,
                base_url=os.getenv("BASE_URL", PROVIDER_BASE_URLS["gemini"]),
                model=os.getenv("CHAT_MODEL", PROVIDER_DEFAULT_MODELS["gemini"]),
            )

        # 4. Bedrock
        if os.getenv("CLAUDE_CODE_USE_BEDROCK") or os.getenv("AWS_BEDROCK_BASE_URL"):
            return cls(
                provider="bedrock",
                api_key=os.getenv("AWS_ACCESS_KEY_ID", "bedrock"),
                base_url=os.getenv("AWS_BEDROCK_BASE_URL", os.getenv("BASE_URL", "")),
                model=os.getenv("CHAT_MODEL", PROVIDER_DEFAULT_MODELS["bedrock"]),
            )

        # 5. Custom
        base_url = os.getenv("BASE_URL", "")
        api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if base_url and api_key:
            return cls(
                provider="custom",
                api_key=api_key,
                base_url=base_url,
                model=os.getenv("CHAT_MODEL", PROVIDER_DEFAULT_MODELS["custom"]),
            )

        # Fallback: no provider
        return cls(
            provider="custom",
            api_key="",
            base_url="http://localhost:11434/v1",
            model=os.getenv("CHAT_MODEL", "llama3"),
        )


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


def _build_http_client() -> httpx.Client:
    """Build httpx client respecting proxy and SSL settings from environment."""
    verify = os.getenv("NODE_TLS_REJECT_UNAUTHORIZED", "1") != "0"
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or None
    timeout = httpx.Timeout(60.0, connect=30.0)
    return httpx.Client(verify=verify, proxy=proxy, timeout=timeout)


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            http_client=_build_http_client(),
        )

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        """Simple chat without tools."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=temperature or self.config.temperature,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Keyword Extraction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a visual research assistant. Given a list of Chinese script lines \
(for a video narration), return a JSON array of concise English image search \
queries — one per input line. Each query should be 2-5 words that would find \
a relevant, visually appealing photo on Wikimedia Commons or a stock photo site.

Rules:
- Output ONLY a JSON array of strings, no explanation.
- The array length MUST equal the number of input lines.
- Prefer concrete, photographable subjects over abstract concepts.
- For greetings or sign-offs, use a relevant cityscape or studio background.
"""


def extract_keywords(lines: list[str], *, config: LLMConfig | None = None) -> list[str]:
    """Return one English image-search query per script line using the configured LLM.

    Args:
        lines: Chinese text lines from the script.
        config: LLM configuration. If None, resolves from environment via LLMConfig.from_env().

    Returns:
        List of English search queries, same length as input.

    Raises:
        RuntimeError: If no LLM provider is configured or the response is malformed.
    """
    if config is None:
        config = LLMConfig.from_env()

    if not config.api_key:
        raise RuntimeError(
            "No LLM provider configured for keyword extraction.\n"
            "Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, "
            "or BASE_URL + API_KEY.\n"
        )

    client = LLMClient(config)

    user_content = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))

    raw = client.chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:200]}")

    # Handle both {"queries": [...]} and bare [...] formats
    if isinstance(parsed, list):
        queries = parsed
    elif isinstance(parsed, dict):
        queries = next((v for v in parsed.values() if isinstance(v, list)), None)
        if queries is None:
            raise RuntimeError(f"LLM response has no array field: {list(parsed.keys())}")
    else:
        raise RuntimeError(f"Unexpected LLM response type: {type(parsed)}")

    if len(queries) != len(lines):
        if len(queries) < len(lines):
            queries.extend(["global finance economy"] * (len(lines) - len(queries)))
        else:
            queries = queries[: len(lines)]

    return [str(q) for q in queries]


if __name__ == "__main__":
    script_path = sys.argv[1] if len(sys.argv) > 1 else "script.txt"
    raw_lines = [
        l.strip()
        for l in Path(script_path).read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    # Strip @@ overrides for extraction
    clean_lines = [l.split("@@")[0].strip() for l in raw_lines]

    keywords = extract_keywords(clean_lines)
    for i, (line, kw) in enumerate(zip(clean_lines, keywords), 1):
        print(f"  [{i}] {line[:30]:30s} → {kw}")
