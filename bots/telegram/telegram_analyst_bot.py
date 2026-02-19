from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analyst_bot.config import Settings
from analyst_bot.data_context import build_dataframe_context
from analyst_bot.deepseek_client import DeepSeekChatConfig, DeepSeekClient
from analyst_bot.fmp_client import FMPClient, FMPClientError
from analyst_bot.router import clean_group_query, extract_symbol, parse_lookback_days, route_intent

try:
    from telegram import Update
    from telegram.constants import ChatAction
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: python-telegram-bot. Install with `pip install python-telegram-bot`."
    ) from e


logger = logging.getLogger("telegram_analyst_bot")


def _load_dotenv(path: str = ".env") -> None:
    candidates = [Path(path)]
    if not Path(path).is_absolute():
        candidates.append(Path(__file__).resolve().parent / path)

    env_path: Path | None = None
    for p in candidates:
        if p.exists() and p.is_file():
            env_path = p
            break

    if env_path is None:
        logger.warning("No .env file found (looked in: %s).", ", ".join(str(p) for p in candidates))
        return

    logger.info("Loaded env file: %s", env_path)
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def _truncate(text: str, limit: int = 3800) -> str:
    t = (text or "").strip()
    return t if len(t) <= limit else t[: limit - 20].rstrip() + "\n\n...(truncated)"


def _is_allowed_chat(update: Update, allowed_chat_id: int | None) -> bool:
    if allowed_chat_id is None:
        return True
    chat = update.effective_chat
    return chat is not None and chat.id == allowed_chat_id


def _should_respond(update: Update, text: str, bot_username: str | None, *, require_mention: bool) -> bool:
    chat = update.effective_chat
    if chat is None:
        return False
    if chat.type == "private":
        return True
    if text.startswith("/ask"):
        return True
    if not require_mention:
        return True
    if not bot_username:
        return False
    return f"@{bot_username}".lower() in text.lower()


async def _handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE, raw_text: str) -> None:
    if update.message is None:
        return

    settings: Settings = context.application.bot_data["settings"]
    llm: DeepSeekClient = context.application.bot_data["llm"]
    data_client = context.application.bot_data["data_client"]

    chat = update.effective_chat
    chat_id = chat.id if chat is not None else None
    if not _is_allowed_chat(update, settings.telegram_chat_id):
        logger.info("Ignoring message from chat_id=%s (allowed=%s)", chat_id, settings.telegram_chat_id)
        return

    bot_username = getattr(context.bot, "username", None)
    if not _should_respond(update, raw_text, bot_username, require_mention=settings.group_require_mention):
        logger.info("Ignoring group message (no mention): chat_id=%s text=%r", chat_id, raw_text[:200])
        return

    question = clean_group_query(raw_text, bot_username)
    if not question:
        await update.message.reply_text("질문을 함께 보내주세요. 예: `/ask TSLA 애널리스트 리비전 추이 알려줘`")
        return

    symbol = extract_symbol(question) or context.chat_data.get("last_symbol")
    if symbol:
        context.chat_data["last_symbol"] = symbol

    intent = route_intent(question)
    lookback_days = parse_lookback_days(question, settings.default_lookback_days)
    end_date = datetime.now(tz=timezone.utc).date()
    start_date = end_date - timedelta(days=lookback_days)

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    if not symbol:
        await update.message.reply_text("심볼(예: TSLA, AAPL)을 질문에 포함해 주세요.")
        return

    try:
        if intent == "analyst_revisions":
            result = await asyncio.to_thread(
                data_client.equity_analyst_revisions,
                symbol,
                start_date=start_date,
                end_date=end_date,
            )
        elif intent == "price_history":
            result = await asyncio.to_thread(
                data_client.equity_price_history,
                symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d",
            )
        else:
            await update.message.reply_text(
                "어떤 데이터를 조회해야 할지 확실하지 않아요. 예: 'TSLA 애널리스트 revision 추이', 'TSLA 주가 흐름' 처럼 질문해 주세요."
            )
            return

        data_context = build_dataframe_context(
            result.df, endpoint=result.endpoint, max_preview_rows=settings.max_preview_rows
        )
        answer = await asyncio.to_thread(
            llm.generate,
            system_prompt=_system_prompt(),
            user_prompt=_build_user_prompt(question, data_context_json=data_context),
        )
        await update.message.reply_text(_truncate(answer))
    except FMPClientError as e:
        err_text = str(e)
        legacy_hint = ""
        if "legacy endpoint" in err_text.lower():
            legacy_hint = (
                "\n\n원인: FMP에서 레거시(/api/v3 등) 엔드포인트를 차단했거나 플랜 제한이 있어요.\n"
                "해결: `stable` 엔드포인트 사용/플랜 업그레이드를 확인해 주세요."
            )
        await update.message.reply_text(
            _truncate(
                "FMP에서 데이터를 가져오지 못했어요.\n"
                f"- 심볼: {symbol or 'N/A'}\n"
                f"- 오류: {e}{legacy_hint}"
            )
        )
    except Exception as e:
        logger.exception("Unhandled error")
        await update.message.reply_text(_truncate(f"처리 중 오류가 발생했어요: {e}"))


def _system_prompt() -> str:
    return (
        "You are a helpful Korean financial analyst assistant.\n"
        "You will be given a user's question and structured market/financial data fetched via Financial Modeling Prep (FMP).\n"
        "Rules:\n"
        "- Use ONLY the provided data. Never invent numbers.\n"
        "- If the data is missing or insufficient, say what you couldn't retrieve and ask a clarifying question.\n"
        "- Keep the answer concise and practical.\n"
        "- End with a short disclaimer: '투자 조언이 아닙니다.'"
    )


def _build_user_prompt(question: str, *, data_context_json: str) -> str:
    return (
        "사용자 질문:\n"
        f"{question}\n\n"
        "API 데이터(JSON):\n"
        f"{data_context_json}\n\n"
        "요청:\n"
        "- 위 데이터만 근거로 한국어로 답변해줘.\n"
        "- 핵심 요약(불릿) + 간단한 해석 + 확인해야 할 포인트(데이터 한계/주의사항)를 포함해줘.\n"
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not _is_allowed_chat(update, settings.telegram_chat_id):
        return
    msg = (
        "안녕하세요. FMP(Financial Modeling Prep) 데이터 기반 애널리스트 봇입니다.\n\n"
        "예시:\n"
        "- 최근 TSLA 애널리스트 revision 추이가 어떤지 알려줘\n"
        "- TSLA 최근 6개월 주가 흐름 요약해줘\n\n"
        "그룹 채팅에서는 `/ask 질문` 또는 봇 멘션(@botname)으로 호출하세요."
    )
    if update.message:
        await update.message.reply_text(msg)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not _is_allowed_chat(update, settings.telegram_chat_id):
        return
    msg = (
        "사용법:\n"
        "- 개인 채팅: 그냥 질문하면 됩니다.\n"
        "- 그룹 채팅: `/ask 질문` 또는 @멘션 포함.\n"
        "- 종목 심볼(예: TSLA, AAPL)을 포함하면 정확도가 올라갑니다.\n"
        "- 기간: '3개월', '1년', 'last 90 days' 같은 표현을 인식합니다.\n\n"
        "명령어:\n"
        "- /start\n"
        "- /help\n"
        "- /reset (최근 심볼 컨텍스트 초기화)"
    )
    if update.message:
        await update.message.reply_text(msg)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    if not _is_allowed_chat(update, settings.telegram_chat_id):
        return
    context.chat_data.clear()
    if update.message:
        await update.message.reply_text("컨텍스트를 초기화했어요. 다시 심볼을 포함해 질문해 주세요.")


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.text is None:
        return
    await _handle_query(update, context, update.message.text.strip())


async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings: Settings = context.application.bot_data["settings"]
    chat = update.effective_chat
    chat_id = chat.id if chat is not None else None
    msg = (
        "OK\n"
        f"- chat_id: {chat_id}\n"
        f"- allowed_chat_id: {settings.telegram_chat_id}\n"
        f"- env: TELEGRAM_BOT_TOKEN={'set' if bool(settings.telegram_bot_token) else 'missing'}, "
        f"DEEPSEEK_API_KEY={'set' if bool(settings.deepseek_api_key) else 'missing'}, "
        f"FMP_API_KEY={'set' if bool(settings.fmp_api_key) else 'missing'}"
    )
    if update.message:
        await update.message.reply_text(msg)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.message.text is None:
        return
    await _handle_query(update, context, update.message.text.strip())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _load_dotenv()
    settings = Settings.from_env()
    os.environ.setdefault("FMP_API_KEY", settings.fmp_api_key)

    data_client = FMPClient(api_key=settings.fmp_api_key, base_url=settings.fmp_base_url)

    llm = DeepSeekClient(
        DeepSeekChatConfig(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            temperature=settings.deepseek_temperature,
            max_tokens=settings.deepseek_max_tokens,
        )
    )

    app = ApplicationBuilder().token(settings.telegram_bot_token).build()
    app.bot_data["settings"] = settings
    app.bot_data["llm"] = llm
    app.bot_data["data_client"] = data_client
    if settings.telegram_chat_id is not None:
        logger.info("Allowed chat id: %s", settings.telegram_chat_id)
    logger.info("Data source: fmp")

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("health", cmd_health))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    logger.info("Bot started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
