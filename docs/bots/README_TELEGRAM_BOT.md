# Telegram 애널리스트 봇 (FMP + DeepSeek)

## 목표
텔레그램에 질문을 입력하면:
1) Financial Modeling Prep(FMP) API로 금융 데이터를 조회하고  
2) DeepSeek 모델이 데이터 기반으로 한국어 답변을 생성합니다.

## 준비물
- Telegram BotFather로 만든 봇 토큰: `TELEGRAM_BOT_TOKEN`
- DeepSeek API 키: `DEEPSEEK_API_KEY`
- Financial Modeling Prep API 키: `FMP_API_KEY`

## 설치
```bash
python -m pip install -r bots/telegram/requirements.txt
```

## 환경변수
`.env.example`를 참고해 `.env`를 만들거나(자동 로드), 아래 값을 직접 환경변수로 설정하세요.
- `TELEGRAM_BOT_TOKEN`
- `DEEPSEEK_API_KEY`
- `FMP_API_KEY`
- (선택) `TELEGRAM_CHAT_ID` (특정 채팅방만 응답하도록 제한)

## 실행
```bash
python bots/telegram/telegram_analyst_bot.py
```

## FMP 키 테스트
FMP가 `/api/v3/*` 같은 레거시 엔드포인트를 막는 경우가 있어요. 아래처럼 `stable` 엔드포인트로 테스트하세요.
- `https://financialmodelingprep.com/stable/profile?symbol=TSLA&apikey=YOUR_KEY`

## 사용 예시
- 개인 채팅: 그냥 질문
  - `최근 TSLA 애널리스트 revision 추이가 어떤지 알려줘`
- 그룹 채팅: `/ask` 또는 봇 멘션(@botname)
  - `/ask TSLA 최근 6개월 주가 흐름 요약해줘`
