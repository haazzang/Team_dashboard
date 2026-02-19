from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class DeepSeekClientError(RuntimeError):
    pass


def _translate_llm_error(err: Exception) -> Exception:
    msg = str(err)
    lower = msg.lower()
    if "invalid_api_key" in lower or "incorrect api key" in lower:
        return DeepSeekClientError("Invalid DeepSeek API key (DEEPSEEK_API_KEY).")
    if "insufficient_balance" in lower or "insufficient balance" in lower or "error code: 402" in lower:
        return DeepSeekClientError("DeepSeek balance/credit is exhausted.")
    if "insufficient_quota" in lower or "exceeded your current quota" in lower:
        return DeepSeekClientError("DeepSeek quota exceeded.")
    return DeepSeekClientError(msg)


@dataclass(frozen=True)
class DeepSeekChatConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_tokens: int = 900


class DeepSeekClient:
    def __init__(self, config: DeepSeekChatConfig) -> None:
        self._config = config

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        cfg = self._config
        try:
            import json

            try:
                import requests  # type: ignore
            except ModuleNotFoundError as e:  # pragma: no cover
                raise DeepSeekClientError(
                    "Missing dependency: `requests`. Install with `pip install requests`."
                ) from e

            base = (cfg.base_url or "").rstrip("/")
            if not base:
                raise DeepSeekClientError("Invalid DeepSeek base_url.")
            url = f"{base}/chat/completions"

            payload: dict[str, Any] = {
                "model": cfg.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
                "stream": False,
            }
            headers = {
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=45)
            if resp.status_code != 200:
                body = (resp.text or "").strip()
                snippet = body[:500] + ("..." if len(body) > 500 else "")
                raise DeepSeekClientError(f"DeepSeek HTTP {resp.status_code}: {snippet}")

            data = resp.json()
            if isinstance(data, dict) and "error" in data and data["error"]:
                raise DeepSeekClientError(str(data["error"]))
            choices = data.get("choices") if isinstance(data, dict) else None
            if not choices or not isinstance(choices, list):
                raise DeepSeekClientError(f"Unexpected DeepSeek response: {data}")
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = message.get("content") if isinstance(message, dict) else None
            return (content or "").strip()
        except Exception as e:
            raise _translate_llm_error(e) from e
