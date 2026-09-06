"""LLM/embedder config for the broker pipeline.

Priority (same as agent):
  1. ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN → Anthropic (via OpenAI-compat)
  2. OPENAI_API_KEY → OpenAI
  3. GEMINI_API_KEY → Gemini (via OpenAI-compat endpoint)
  4. BASE_URL + API_KEY → custom / local (e.g. Ollama, vLLM)
  5. Fallback → localhost:8081/v1 with model qwen3-8b

Embedder env vars (separate from the chat LLM):
  EMBEDDING_BASE_URL   – defaults to BASE_URL or localhost:8082/v1
  EMBEDDING_MODEL      – defaults to qwen3-8b-embedding
  EMBEDDING_API_KEY    – defaults to API_KEY / OPENAI_API_KEY / sk-xxx

Vector-store env vars:
  VECTOR_STORE_PATH        – defaults to db/faiss_db
  VECTOR_STORE_COLLECTION  – defaults to telemem
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Provider look-up tables (mirrors agent/config.py)
# ---------------------------------------------------------------------------

_BASE_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "local": "qwen3-8b",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_llm() -> dict:
    """Return a telemem-compatible ``llm`` section dict."""
    # 1. Anthropic
    key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    if key:
        base_url = os.getenv("ANTHROPIC_BASE_URL", _BASE_URLS["anthropic"])
        model = os.getenv("CHAT_MODEL", _DEFAULT_MODELS["anthropic"])
        return {
            "provider": "openai",
            "config": {"model": model, "openai_base_url": base_url, "api_key": key},
        }

    # 2. OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if key:
        base_url = os.getenv("BASE_URL", _BASE_URLS["openai"])
        model = os.getenv("CHAT_MODEL", _DEFAULT_MODELS["openai"])
        return {
            "provider": "openai",
            "config": {"model": model, "openai_base_url": base_url, "api_key": key},
        }

    # 3. Gemini (OpenAI-compatible endpoint)
    key = os.getenv("GEMINI_API_KEY")
    if key:
        base_url = os.getenv("BASE_URL", _BASE_URLS["gemini"])
        model = os.getenv("CHAT_MODEL", _DEFAULT_MODELS["gemini"])
        return {
            "provider": "openai",
            "config": {"model": model, "openai_base_url": base_url, "api_key": key},
        }

    # 4. Custom / local (BASE_URL + API_KEY)
    base_url = os.getenv("BASE_URL", "http://localhost:8081/v1")
    key = os.getenv("API_KEY", "sk-xxx")
    model = os.getenv("CHAT_MODEL", _DEFAULT_MODELS["local"])
    return {
        "provider": "openai",
        "config": {"model": model, "openai_base_url": base_url, "api_key": key},
    }


def _ensure_v1(url: str) -> str:
    """Append /v1 if the URL doesn't already end with it."""
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


def _resolve_embedder() -> dict:
    """Return a telemem-compatible ``embedder`` section dict."""
    # Embedding endpoint can differ from the chat endpoint (separate server).
    base_url = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("BASE_URL", "http://localhost:8082/v1")
    base_url = _ensure_v1(base_url)
    key = (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("ANTHROPIC_AUTH_TOKEN")
        or os.getenv("API_KEY", "sk-xxx")
    )
    model = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-0.6B")
    dims = int(os.getenv("EMBEDDING_DIMS", "1024"))
    return {
        "provider": "openai",
        "config": {"model": model, "openai_base_url": base_url, "api_key": key, "embedding_dims": dims},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_telemem_config() -> dict:
    """Return a TeleMemoryConfig-compatible dict built from environment variables."""
    return {
        "llm": _resolve_llm(),
        "embedder": _resolve_embedder(),
        "vector_store": {
            "provider": "faiss",
            "config": {
                "collection_name": os.getenv("VECTOR_STORE_COLLECTION", "telemem"),
                "path": os.getenv("VECTOR_STORE_PATH", "db/faiss_db"),
                "embedding_model_dims": int(os.getenv("EMBEDDING_DIMS", "1024")),
            },
        },
        "history_db_path": os.getenv("HISTORY_DB_PATH", "db/history.db"),
        "buffer_size": int(os.getenv("BUFFER_SIZE", "64")),
        "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.95")),
    }


def _maybe_disable_ssl_verify() -> None:
    """Patch ssl.create_default_context to skip cert verification.

    Activated by NODE_TLS_REJECT_UNAUTHORIZED=0 (same env var as agent/llm.py).
    Needed when the network proxy uses a self-signed certificate.
    """
    if os.getenv("NODE_TLS_REJECT_UNAUTHORIZED", "1") != "0":
        return
    import ssl
    _orig = ssl.create_default_context

    def _patched(*args, **kwargs):
        ctx = _orig(*args, **kwargs)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    ssl.create_default_context = _patched


def _patch_memory_ssl(memory) -> None:
    """Replace the embedding (and LLM) client's internal http client with one
    that skips SSL verification.

    mem0's OpenAIEmbedding creates OpenAI(api_key, base_url) without a custom
    http_client, so there is no way to inject verify=False from outside except
    by replacing the client after construction.

    Only active when NODE_TLS_REJECT_UNAUTHORIZED=0.
    """
    if os.getenv("NODE_TLS_REJECT_UNAUTHORIZED", "1") != "0":
        return

    import httpx2
    from openai import OpenAI

    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or None
    http_client = httpx2.Client(verify=False, proxy=proxy)

    emb = getattr(memory, "embedding_model", None)
    if emb and hasattr(emb, "client"):
        emb.client = OpenAI(
            api_key=emb.client.api_key,
            base_url=str(emb.client.base_url),
            http_client=http_client,
        )


def make_memory(config_path: str | None = None):
    """Return a TeleMemory instance configured from a file or env vars.

    This is shared by ingest_channel.py and query_memory.py.
    """
    _maybe_disable_ssl_verify()

    import telemem as mem0
    from telemem.configs import TeleMemoryConfig

    if config_path:
        from telemem.utils import load_config
        memory = mem0.Memory(config=load_config(config_path))
    else:
        memory = mem0.Memory(config=TeleMemoryConfig(**build_telemem_config()))

    _patch_memory_ssl(memory)
    return memory
