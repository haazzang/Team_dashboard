"""
Gmail Push Notification Webhook Server

Gmail에서 새 메일이 도착하면 자동으로 Swap Report를 처리합니다.

설정 방법:
1. Google Cloud Console에서 Pub/Sub 설정
2. ngrok 또는 서버로 webhook 엔드포인트 노출
3. 이 서버 실행: python automation/swap/gmail_webhook_server.py
"""

import os
import json
import base64
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify

# swap_report_fetcher 모듈 import
try:
    from .swap_report_fetcher import (
        get_gmail_service,
        init_database,
        process_message,
        parse_excel_report,
        DB_FILE,
        MAIL_SUBJECT,
    )
except ImportError:
    from swap_report_fetcher import (
        get_gmail_service,
        init_database,
        process_message,
        parse_excel_report,
        DB_FILE,
        MAIL_SUBJECT,
    )

app = Flask(__name__)

# 로그 파일
LOG_FILE = Path(__file__).resolve().parent / 'webhook_server.log'


def log_message(message):
    """로그 메시지 기록"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + '\n')


def process_new_emails():
    """새 이메일 처리"""
    try:
        log_message("새 메일 확인 시작...")

        # DB 연결
        conn = init_database()

        # Gmail 서비스
        service = get_gmail_service()

        # 최근 메일 검색 (최근 10개만)
        query = f'subject:"{MAIL_SUBJECT}" has:attachment'
        results = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=10
        ).execute()

        messages = results.get('messages', [])
        log_message(f"검색된 메일: {len(messages)}개")

        total_processed = 0

        for msg in messages:
            downloaded_files = process_message(service, msg['id'], conn)

            for filepath, filename, message_id in downloaded_files:
                if parse_excel_report(filepath, filename, message_id, conn):
                    total_processed += 1
                    log_message(f"처리 완료: {filename}")

        conn.close()
        log_message(f"총 {total_processed}개 리포트 처리됨")

        return total_processed

    except Exception as e:
        log_message(f"오류 발생: {e}")
        return 0


@app.route('/webhook/gmail', methods=['POST'])
def gmail_webhook():
    """Gmail Push Notification 수신 엔드포인트"""
    try:
        # Pub/Sub 메시지 파싱
        envelope = request.get_json()

        if not envelope:
            log_message("빈 요청 수신")
            return jsonify({'status': 'empty'}), 200

        # Pub/Sub 메시지 디코딩
        if 'message' in envelope:
            message = envelope['message']
            if 'data' in message:
                data = base64.b64decode(message['data']).decode('utf-8')
                notification = json.loads(data)
                log_message(f"Gmail 알림 수신: {notification}")

                # 비동기로 메일 처리
                thread = threading.Thread(target=process_new_emails)
                thread.start()

        return jsonify({'status': 'ok'}), 200

    except Exception as e:
        log_message(f"Webhook 오류: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/webhook/manual', methods=['POST', 'GET'])
def manual_trigger():
    """수동 트리거 엔드포인트"""
    log_message("수동 트리거 요청")
    processed = process_new_emails()
    return jsonify({
        'status': 'ok',
        'processed': processed
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/status', methods=['GET'])
def status():
    """현재 상태 확인"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM reports')
        total_reports = cursor.fetchone()[0]

        cursor.execute('SELECT MAX(report_date) FROM reports')
        latest_date = cursor.fetchone()[0]

        conn.close()

        return jsonify({
            'total_reports': total_reports,
            'latest_report_date': latest_date,
            'db_file': str(DB_FILE)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def setup_gmail_watch(service):
    """Gmail Watch 설정 (Pub/Sub 토픽 구독)"""
    try:
        # 이 부분은 Google Cloud Console에서 Pub/Sub 토픽 생성 후 설정
        # 프로젝트 ID와 토픽 이름을 환경 변수로 설정
        project_id = os.environ.get('GOOGLE_CLOUD_PROJECT', 'your-project-id')
        topic_name = os.environ.get('PUBSUB_TOPIC', 'gmail-notifications')

        request_body = {
            'labelIds': ['INBOX'],
            'topicName': f'projects/{project_id}/topics/{topic_name}'
        }

        response = service.users().watch(userId='me', body=request_body).execute()
        log_message(f"Gmail Watch 설정 완료: {response}")
        return response

    except Exception as e:
        log_message(f"Gmail Watch 설정 실패: {e}")
        return None


if __name__ == '__main__':
    log_message("=" * 50)
    log_message("Gmail Webhook Server 시작")
    log_message("=" * 50)

    # 서버 시작 (포트 5000)
    # 프로덕션에서는 gunicorn 사용 권장
    app.run(host='0.0.0.0', port=5000, debug=False)
