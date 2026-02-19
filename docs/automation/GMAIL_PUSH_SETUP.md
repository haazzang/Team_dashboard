# Gmail Push Notification 설정 가이드

메일이 도착하면 자동으로 Swap Report를 DB에 저장하는 시스템입니다.

## 1. Google Cloud Pub/Sub 설정

### 1.1 Pub/Sub 토픽 생성
```bash
# Google Cloud Console에서 또는 gcloud CLI로:
gcloud pubsub topics create gmail-notifications
```

### 1.2 Gmail API에 Pub/Sub 권한 부여
```bash
# Gmail API 서비스 계정에 Publisher 권한 부여
gcloud pubsub topics add-iam-policy-binding gmail-notifications \
    --member="serviceAccount:gmail-api-push@system.gserviceaccount.com" \
    --role="roles/pubsub.publisher"
```

### 1.3 Pub/Sub 구독 생성 (Push 방식)
```bash
# YOUR_WEBHOOK_URL을 실제 서버 URL로 변경
gcloud pubsub subscriptions create gmail-push-sub \
    --topic=gmail-notifications \
    --push-endpoint=https://YOUR_WEBHOOK_URL/webhook/gmail
```

## 2. 서버 실행

### 2.1 로컬 테스트 (ngrok 사용)
```bash
# 터미널 1: 서버 실행
pip install flask
python automation/swap/gmail_webhook_server.py

# 터미널 2: ngrok으로 외부 노출
ngrok http 5000
# 표시된 https URL을 Pub/Sub 구독 push-endpoint로 설정
```

### 2.2 프로덕션 배포 옵션

#### Option A: Railway/Render 배포
1. GitHub에 push
2. Railway 또는 Render에서 자동 배포
3. 환경 변수 설정:
   - `GOOGLE_CLOUD_PROJECT`: GCP 프로젝트 ID
   - `PUBSUB_TOPIC`: gmail-notifications

#### Option B: Google Cloud Run 배포
```bash
# Dockerfile 생성 후
gcloud run deploy gmail-webhook \
    --source . \
    --region=asia-northeast3 \
    --allow-unauthenticated
```

## 3. Gmail Watch 활성화

서버가 실행된 후, Gmail Watch를 활성화해야 합니다:

```python
# Python 콘솔에서 실행
from automation.swap.gmail_webhook_server import get_gmail_service, setup_gmail_watch
import os

os.environ['GOOGLE_CLOUD_PROJECT'] = 'your-project-id'
os.environ['PUBSUB_TOPIC'] = 'gmail-notifications'

service = get_gmail_service()
setup_gmail_watch(service)
```

또는 API 엔드포인트로:
```bash
curl -X POST https://YOUR_SERVER/webhook/manual
```

## 4. 엔드포인트

| 엔드포인트 | 메소드 | 설명 |
|-----------|--------|------|
| `/webhook/gmail` | POST | Pub/Sub 알림 수신 |
| `/webhook/manual` | GET/POST | 수동으로 새 메일 확인 |
| `/health` | GET | 헬스 체크 |
| `/status` | GET | DB 상태 확인 |

## 5. 로그 확인

```bash
# 로그 파일 확인
tail -f webhook_server.log
```

## 6. 간단한 대안: Cron 스케줄링

Pub/Sub 설정이 복잡하다면, 간단히 주기적으로 체크하는 방식도 가능합니다:

```bash
# Mac: crontab -e
# 매일 오후 7시에 실행
0 19 * * * cd /Users/hyejinha/Desktop/Workspace/Team && /usr/bin/python3 automation/swap/swap_report_fetcher.py >> cron.log 2>&1

# 또는 30분마다 실행
*/30 * * * * cd /Users/hyejinha/Desktop/Workspace/Team && /usr/bin/python3 automation/swap/swap_report_fetcher.py >> cron.log 2>&1
```

## 7. 문제 해결

### Watch 만료
Gmail Watch는 7일 후 만료됩니다. 자동 갱신을 위해:
```python
# 매일 실행되는 cron job 추가
0 0 * * * python -c "from automation.swap.gmail_webhook_server import *; setup_gmail_watch(get_gmail_service())"
```

### 인증 오류
- `credentials.json` 파일 확인
- `token.json` 삭제 후 재인증
