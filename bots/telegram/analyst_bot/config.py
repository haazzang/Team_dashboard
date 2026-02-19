from __future__ import annotations

import os
from dataclasses import dataclass


def _getenv(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _getenv_int(name: str, default: int) -> int:
    raw = _getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _getenv_bool(name: str, default: bool) -> bool:
    raw = _getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    deepseek_api_key: str
    fmp_api_key: str
    telegram_chat_id: int | None = None

    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_temperature: float = 0.2
    deepseek_max_tokens: int = 900

    fmp_base_url: str = "https://financialmodelingprep.com"
    default_lookback_days: int = 180
    max_preview_rows: int = 12

    group_require_mention: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        telegram_bot_token = _getenv("TELEGRAM_BOT_TOKEN")
        deepseek_api_key = _getenv("DEEPSEEK_API_KEY")
        fmp_api_key = _getenv("FMP_API_KEY")
        telegram_chat_id_raw = _getenv("TELEGRAM_CHAT_ID")
        if not telegram_bot_token:
            raise RuntimeError("Missing env var: TELEGRAM_BOT_TOKEN")
        if not deepseek_api_key:
            raise RuntimeError("Missing env var: DEEPSEEK_API_KEY")
        if not fmp_api_key:
            raise RuntimeError("Missing env var: FMP_API_KEY")

        telegram_chat_id: int | None = None
        if telegram_chat_id_raw is not None:
            try:
                telegram_chat_id = int(telegram_chat_id_raw)
            except ValueError as e:
                raise RuntimeError("Invalid TELEGRAM_CHAT_ID (must be an integer).") from e

        base_url = _getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com") or "https://api.deepseek.com"
        model = _getenv("DEEPSEEK_MODEL", "deepseek-chat") or "deepseek-chat"
        temperature_raw = _getenv("DEEPSEEK_TEMPERATURE")
        try:
            temperature = float(temperature_raw) if temperature_raw is not None else 0.2
        except ValueError:
            temperature = 0.2

        return cls(
            telegram_bot_token=telegram_bot_token,
            deepseek_api_key=deepseek_api_key,
            fmp_api_key=fmp_api_key,
            telegram_chat_id=telegram_chat_id,
            deepseek_base_url=base_url,
            deepseek_model=model,
            deepseek_temperature=temperature,
            deepseek_max_tokens=_getenv_int("DEEPSEEK_MAX_TOKENS", 900),
            fmp_base_url=_getenv("FMP_BASE_URL", "https://financialmodelingprep.com") or "https://financialmodelingprep.com",
            default_lookback_days=_getenv_int("DEFAULT_LOOKBACK_DAYS", 180),
            max_preview_rows=_getenv_int("MAX_PREVIEW_ROWS", 12),
            group_require_mention=_getenv_bool("GROUP_REQUIRE_MENTION", True),
        )
