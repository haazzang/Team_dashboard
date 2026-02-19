"""
Swap Report Fetcher - Gmail에서 Swap Report 첨부파일을 다운로드하고 SQLite DB에 저장

사용 전 설정:
1. Google Cloud Console (https://console.cloud.google.com) 접속
2. 새 프로젝트 생성 또는 기존 프로젝트 선택
3. Gmail API 활성화: APIs & Services > Library > Gmail API > Enable
4. OAuth 2.0 설정: APIs & Services > Credentials > Create Credentials > OAuth client ID
   - Application type: Desktop app
   - Download JSON 클릭 후 파일을 'credentials.json'으로 저장
5. OAuth consent screen 설정: APIs & Services > OAuth consent screen
   - User Type: External (또는 Internal if Workspace)
   - App name, email 입력
   - Scopes에 'https://www.googleapis.com/auth/gmail.readonly' 추가
6. credentials.json 파일을 Team 루트 폴더에 저장

실행:
    python automation/swap/swap_report_fetcher.py
"""

import os
import base64
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import re

# Google API imports
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    print("Google API 라이브러리가 설치되어 있지 않습니다.")
    print("설치: pip install google-auth google-auth-oauthlib google-api-python-client")

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# 설정
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
CREDENTIALS_FILE = ROOT_DIR / 'credentials.json'
TOKEN_FILE = ROOT_DIR / 'token.json'
DB_FILE = ROOT_DIR / 'swap_reports.db'
DOWNLOAD_DIR = ROOT_DIR / 'swap_reports'

# 검색할 메일 제목
MAIL_SUBJECT = 'FW: JMLNKWGE Synthetic Portfolio EOD Report'
# 시작 파일명 (이 날짜 이후의 파일만 처리)
START_FILENAME = '20260119_1800_JMLNKWGE_Report.xlsx'


def get_gmail_service():
    """Gmail API 서비스 객체 생성"""
    if not GOOGLE_API_AVAILABLE:
        raise ImportError("Google API 라이브러리가 설치되어 있지 않습니다.")

    creds = None

    # 기존 토큰 확인
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    # 토큰이 없거나 만료된 경우
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"credentials.json 파일이 없습니다.\n"
                    f"위치: {CREDENTIALS_FILE}\n"
                    f"Google Cloud Console에서 OAuth 2.0 credentials를 다운로드하세요."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        # 토큰 저장
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)


def init_database():
    """SQLite 데이터베이스 초기화"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 메인 리포트 정보 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date DATE NOT NULL,
            filename TEXT NOT NULL UNIQUE,
            email_id TEXT,
            downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Overview 시트 데이터
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS overview (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER,
            field_name TEXT,
            field_value TEXT,
            FOREIGN KEY (report_id) REFERENCES reports(id)
        )
    ''')

    # Underlying 시트 데이터 (개별 종목)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS underlying (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER,
            ticker TEXT,
            name TEXT,
            quantity REAL,
            price REAL,
            market_value_usd REAL,
            weight REAL,
            pnl_usd REAL,
            pnl_pct REAL,
            contribution REAL,
            sector TEXT,
            country TEXT,
            currency TEXT,
            FOREIGN KEY (report_id) REFERENCES reports(id)
        )
    ''')

    # Und 시트 데이터 (요약/추가 정보)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS und_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER,
            category TEXT,
            field_name TEXT,
            field_value TEXT,
            FOREIGN KEY (report_id) REFERENCES reports(id)
        )
    ''')

    # 일별 포트폴리오 요약
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER,
            report_date DATE,
            total_nav_usd REAL,
            daily_pnl_usd REAL,
            daily_return_pct REAL,
            mtd_return_pct REAL,
            ytd_return_pct REAL,
            FOREIGN KEY (report_id) REFERENCES reports(id)
        )
    ''')

    conn.commit()
    return conn


def search_emails(service, subject=MAIL_SUBJECT, max_results=100):
    """Gmail에서 특정 제목의 메일 검색"""
    query = f'subject:"{subject}" has:attachment'

    results = service.users().messages().list(
        userId='me',
        q=query,
        maxResults=max_results
    ).execute()

    messages = results.get('messages', [])
    return messages


def get_attachment(service, message_id, attachment_id):
    """첨부파일 다운로드"""
    attachment = service.users().messages().attachments().get(
        userId='me',
        messageId=message_id,
        id=attachment_id
    ).execute()

    data = attachment['data']
    file_data = base64.urlsafe_b64decode(data)
    return file_data


def extract_date_from_filename(filename):
    """파일명에서 날짜 추출 (예: 20260119_1800_JMLNKWGE_Report.xlsx)"""
    match = re.match(r'(\d{8})_\d{4}_', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d').date()
    return None


def process_message(service, message_id, conn):
    """메일 메시지 처리 및 첨부파일 다운로드"""
    message = service.users().messages().get(
        userId='me',
        id=message_id,
        format='full'
    ).execute()

    # 첨부파일 확인
    parts = message.get('payload', {}).get('parts', [])

    downloaded_files = []

    for part in parts:
        filename = part.get('filename', '')
        if filename.endswith('.xlsx') and 'JMLNKWGE' in filename and 'Report' in filename:
            # 시작 파일명 이후인지 확인
            if filename < START_FILENAME:
                print(f"스킵 (시작일 이전): {filename}")
                continue

            # 이미 DB에 있는지 확인
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM reports WHERE filename = ?', (filename,))
            if cursor.fetchone():
                print(f"스킵 (이미 존재): {filename}")
                continue

            # 첨부파일 다운로드
            attachment_id = part['body'].get('attachmentId')
            if attachment_id:
                file_data = get_attachment(service, message_id, attachment_id)

                # 다운로드 폴더 생성
                DOWNLOAD_DIR.mkdir(exist_ok=True)

                # 파일 저장
                filepath = DOWNLOAD_DIR / filename
                with open(filepath, 'wb') as f:
                    f.write(file_data)

                print(f"다운로드 완료: {filename}")
                downloaded_files.append((filepath, filename, message_id))

    return downloaded_files


def parse_excel_report(filepath, filename, message_id, conn):
    """Excel 파일 파싱 및 DB 저장"""
    try:
        # 리포트 날짜 추출
        report_date = extract_date_from_filename(filename)
        if not report_date:
            print(f"날짜 추출 실패: {filename}")
            return False

        # reports 테이블에 삽입
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO reports (report_date, filename, email_id)
            VALUES (?, ?, ?)
        ''', (report_date, filename, message_id))
        report_id = cursor.lastrowid

        # Excel 파일 읽기
        xlsx = pd.ExcelFile(filepath)
        sheet_names = xlsx.sheet_names
        print(f"시트 목록: {sheet_names}")

        # Overview 시트 파싱
        if 'Overview' in sheet_names or 'overview' in [s.lower() for s in sheet_names]:
            overview_sheet = next(s for s in sheet_names if s.lower() == 'overview')
            df_overview = pd.read_excel(xlsx, sheet_name=overview_sheet, header=None)

            for idx, row in df_overview.iterrows():
                if len(row) >= 2 and pd.notna(row[0]):
                    cursor.execute('''
                        INSERT INTO overview (report_id, field_name, field_value)
                        VALUES (?, ?, ?)
                    ''', (report_id, str(row[0]), str(row[1]) if len(row) > 1 and pd.notna(row[1]) else None))

        # Underlying 시트 파싱
        underlying_sheet_names = ['Underlying', 'underlying', 'Component Underlying', 'Component Underlyings']
        for sheet_name in sheet_names:
            if any(u.lower() in sheet_name.lower() for u in underlying_sheet_names):
                # 헤더가 Row 3에 있음 (0-indexed)
                df_underlying = pd.read_excel(xlsx, sheet_name=sheet_name, header=3)

                # 컬럼명 정리 (소문자로 변환하여 매핑)
                df_underlying.columns = [str(c).strip() for c in df_underlying.columns]

                for idx, row in df_underlying.iterrows():
                    # 데이터 추출 (실제 컬럼명: Quantity, Currency, Name, Country of issuer, Current Price,
                    # Market Value (local), Market Value (USD), Sedol Code, RIC Code, etc.)
                    # RIC Code에서 ticker 추출 (예: AMD.OQ -> AMD)
                    ric_code = row.get('RIC Code', '')
                    ticker = str(ric_code).split('.')[0] if pd.notna(ric_code) else ''
                    name = row.get('Name', row.get('name', row.get('Security Name', '')))
                    quantity = row.get('Quantity', 0)
                    price = row.get('Current Price', row.get('Price', 0))
                    market_value = row.get('Market Value (USD)', row.get('Market Value', 0))
                    weight = row.get('Weight', row.get('Weight (%)', 0))
                    pnl = row.get('PnL (USD)', row.get('P&L', row.get('PnL', 0)))
                    pnl_pct = row.get('PnL (%)', row.get('P&L %', row.get('Return', 0)))
                    contribution = row.get('Contribution', row.get('Contrib', 0))
                    sector = row.get('Industry Group', row.get('Sector', ''))
                    country = row.get('Country of issuer', row.get('Country', ''))
                    currency = row.get('Currency', 'USD')

                    if pd.notna(ticker) and str(ticker).strip():
                        cursor.execute('''
                            INSERT INTO underlying (
                                report_id, ticker, name, quantity, price, market_value_usd,
                                weight, pnl_usd, pnl_pct, contribution, sector, country, currency
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            report_id, str(ticker), str(name) if pd.notna(name) else '',
                            float(quantity) if pd.notna(quantity) else 0,
                            float(price) if pd.notna(price) else 0,
                            float(market_value) if pd.notna(market_value) else 0,
                            float(weight) if pd.notna(weight) else 0,
                            float(pnl) if pd.notna(pnl) else 0,
                            float(pnl_pct) if pd.notna(pnl_pct) else 0,
                            float(contribution) if pd.notna(contribution) else 0,
                            str(sector) if pd.notna(sector) else '',
                            str(country) if pd.notna(country) else '',
                            str(currency) if pd.notna(currency) else 'USD'
                        ))
                break

        # Und 시트 파싱 - PNL 데이터 추출
        for sheet_name in sheet_names:
            if sheet_name.lower() == 'und':
                # 헤더가 Row 3에 있음
                df_und = pd.read_excel(xlsx, sheet_name=sheet_name, header=3)
                df_und.columns = [str(c).strip() for c in df_und.columns]

                # PNL 데이터를 underlying 테이블에 업데이트
                for idx, row in df_und.iterrows():
                    ticker_raw = row.get('TICKER', '')
                    if pd.notna(ticker_raw) and str(ticker_raw).strip():
                        # 티커 추출 (예: "AMD UW Equity" -> "AMD", "002896 CS Equity" -> "002896")
                        ticker = str(ticker_raw).split()[0] if pd.notna(ticker_raw) else ''

                        # PNL 값 추출
                        pnl_col = None
                        for col in df_und.columns:
                            if 'TRADE_PNL' in col.upper() or 'PNL' in col.upper():
                                pnl_col = col
                                break

                        if pnl_col:
                            pnl_value = row.get(pnl_col, 0)
                            pnl_value = float(pnl_value) if pd.notna(pnl_value) else 0

                            # END_VALUE로 수익률 계산
                            end_value_col = None
                            for col in df_und.columns:
                                if 'END_VALUE' in col.upper():
                                    end_value_col = col
                                    break

                            end_value = row.get(end_value_col, 0) if end_value_col else 0
                            end_value = float(end_value) if pd.notna(end_value) else 0

                            # 수익률 계산 (PNL / (END_VALUE - PNL) * 100)
                            base_value = end_value - pnl_value
                            pnl_pct = (pnl_value / base_value * 100) if base_value != 0 else 0

                            # underlying 테이블 업데이트
                            cursor.execute('''
                                UPDATE underlying
                                SET pnl_usd = ?, pnl_pct = ?
                                WHERE report_id = ? AND ticker = ?
                            ''', (pnl_value, pnl_pct, report_id, ticker))

        conn.commit()
        print(f"DB 저장 완료: {filename}")
        return True

    except Exception as e:
        print(f"파싱 오류 ({filename}): {e}")
        conn.rollback()
        return False


def fetch_and_store_reports():
    """메인 함수: Gmail에서 리포트를 가져와서 DB에 저장"""
    print("=" * 60)
    print("Swap Report Fetcher")
    print("=" * 60)

    # DB 초기화
    conn = init_database()
    print(f"DB 파일: {DB_FILE}")

    try:
        # Gmail 서비스 연결
        print("\nGmail API 연결 중...")
        service = get_gmail_service()
        print("Gmail API 연결 성공!")

        # 메일 검색
        print(f"\n메일 검색 중... (제목: {MAIL_SUBJECT})")
        messages = search_emails(service)
        print(f"찾은 메일 수: {len(messages)}")

        # 각 메일 처리
        total_downloaded = 0
        total_processed = 0

        for msg in messages:
            downloaded_files = process_message(service, msg['id'], conn)

            for filepath, filename, message_id in downloaded_files:
                total_downloaded += 1
                if parse_excel_report(filepath, filename, message_id, conn):
                    total_processed += 1

        print("\n" + "=" * 60)
        print(f"완료!")
        print(f"다운로드: {total_downloaded}개")
        print(f"DB 저장: {total_processed}개")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n오류: {e}")
        print("\n[설정 방법]")
        print("1. Google Cloud Console (https://console.cloud.google.com) 접속")
        print("2. 프로젝트 생성 후 Gmail API 활성화")
        print("3. OAuth 2.0 credentials 생성 (Desktop app)")
        print("4. credentials.json 다운로드 후 Team 루트 폴더에 저장")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

    finally:
        conn.close()


def show_db_stats():
    """DB 통계 출력"""
    if not DB_FILE.exists():
        print("DB 파일이 없습니다. 먼저 fetch_and_store_reports()를 실행하세요.")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    print("\n[DB 통계]")

    cursor.execute('SELECT COUNT(*) FROM reports')
    print(f"총 리포트 수: {cursor.fetchone()[0]}")

    cursor.execute('SELECT MIN(report_date), MAX(report_date) FROM reports')
    min_date, max_date = cursor.fetchone()
    print(f"기간: {min_date} ~ {max_date}")

    cursor.execute('SELECT COUNT(*) FROM underlying')
    print(f"Underlying 레코드 수: {cursor.fetchone()[0]}")

    conn.close()


if __name__ == '__main__':
    fetch_and_store_reports()
    show_db_stats()
