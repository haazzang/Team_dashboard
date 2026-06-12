"""
Swap Report Fetcher - Gmail(IMAP)에서 Swap Report 첨부파일을 다운로드하고 SQLite DB에 저장

인증 방식: Gmail 앱 비밀번호(IMAP). OAuth 토큰처럼 주기적으로 만료되지 않아
headless 자동화에 안정적입니다.

사용 전 설정 (최초 1회):
1. Gmail 계정에 2단계 인증(2FA) 활성화
   https://myaccount.google.com/security
2. 앱 비밀번호 생성: https://myaccount.google.com/apppasswords
   - 앱 이름 예: "swap-report-fetcher" -> 16자리 비밀번호 발급
3. 프로젝트 루트의 .env 파일에 아래 두 줄 추가:
       GMAIL_USER=your_account@gmail.com
       GMAIL_APP_PASSWORD=xxxxxxxxxxxxxxxx   # 공백 없이 16자
   (환경변수로 직접 export 해도 됩니다.)

실행:
    python automation/swap/swap_report_fetcher.py
"""

import os
import email
import imaplib
import shutil
import sqlite3
import pandas as pd
from email.header import decode_header as _decode_header
from datetime import datetime
from pathlib import Path
import re

# 설정
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
ENV_FILE = ROOT_DIR / '.env'
DB_FILE = ROOT_DIR / 'swap_reports.db'
DOWNLOAD_DIR = ROOT_DIR / 'swap_reports'

# SQLite 작업용 로컬 캐시 경로.
# ROOT_DIR(Desktop)이 iCloud로 동기화되면 쓰기 도중 "disk I/O error"가 나므로,
# 동기화되지 않는 로컬 캐시에서 작업한 뒤 마지막에 DB_FILE 로 복사한다.
WORK_DIR = Path.home() / 'Library' / 'Caches' / 'swap_report_fetcher'
WORK_DB = WORK_DIR / 'swap_reports.db'

# IMAP 설정
IMAP_HOST = 'imap.gmail.com'
IMAP_PORT = 993
MAILBOX = 'INBOX'
GMAIL_USER_ENV = 'GMAIL_USER'
GMAIL_PASSWORD_ENV = 'GMAIL_APP_PASSWORD'

# 검색할 메일 제목 (IMAP SUBJECT 검색은 부분 일치 -> "FW:" 접두사가 있어도 매칭됨)
MAIL_SUBJECT = 'JMLNKWGE Synthetic Portfolio EOD Report'
# 이 날짜(YYYY-MM-DD) 이후 도착한 메일만 검색
SEARCH_SINCE = '2026-01-19'
# 시작 파일명 (이 날짜 이후의 파일만 처리)
START_FILENAME = '20260119_1800_JMLNKWGE_Report.xlsx'


def load_env_file():
    """프로젝트 루트의 .env 를 읽어 os.environ 에 주입 (이미 설정된 값은 유지)."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _decode_filename(raw_name: str) -> str:
    """RFC2047 인코딩된 첨부파일명 디코드."""
    if not raw_name:
        return ''
    parts = _decode_header(raw_name)
    out = ''
    for text, enc in parts:
        if isinstance(text, bytes):
            out += text.decode(enc or 'utf-8', errors='ignore')
        else:
            out += text
    return out


def get_imap_connection():
    """Gmail IMAP 연결 (앱 비밀번호 사용)."""
    user = os.environ.get(GMAIL_USER_ENV, '').strip()
    password = os.environ.get(GMAIL_PASSWORD_ENV, '').strip().replace(' ', '')
    if not user or not password:
        raise RuntimeError(
            f"{GMAIL_USER_ENV} / {GMAIL_PASSWORD_ENV} 환경변수가 설정되어 있지 않습니다.\n"
            f".env 파일에 Gmail 계정과 앱 비밀번호를 추가하세요.\n"
            f"앱 비밀번호 생성: https://myaccount.google.com/apppasswords"
        )
    imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    imap.login(user, password)
    imap.select(MAILBOX)
    return imap


def init_database():
    """SQLite 데이터베이스 초기화 (로컬 캐시에서 작업)."""
    conn = sqlite3.connect(WORK_DB, timeout=60)
    conn.execute('PRAGMA busy_timeout=60000')
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


def search_email_uids(imap, subject=MAIL_SUBJECT, since=SEARCH_SINCE):
    """Gmail IMAP에서 특정 제목 + 날짜 이후의 메일 UID 목록 검색"""
    since_imap = datetime.strptime(since, '%Y-%m-%d').strftime('%d-%b-%Y')
    status, data = imap.uid('search', None, f'(SINCE {since_imap} SUBJECT "{subject}")')
    if status != 'OK':
        return []
    return data[0].split()


def extract_date_from_filename(filename):
    """파일명에서 날짜 추출 (예: 20260119_1800_JMLNKWGE_Report.xlsx)"""
    match = re.match(r'(\d{8})_\d{4}_', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d').date()
    return None


def process_message(imap, uid, conn):
    """메일 메시지(UID) 처리 및 첨부파일 다운로드"""
    uid_text = uid.decode('ascii', errors='ignore') if isinstance(uid, bytes) else str(uid)

    status, msg_data = imap.uid('fetch', uid, '(RFC822)')
    if status != 'OK':
        return []
    raw = next((part[1] for part in msg_data if isinstance(part, tuple)), None)
    if not raw:
        return []
    message = email.message_from_bytes(raw)

    downloaded_files = []

    for part in message.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        filename = _decode_filename(part.get_filename() or '')
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
            payload = part.get_payload(decode=True)
            if payload:
                # 다운로드 폴더 생성
                DOWNLOAD_DIR.mkdir(exist_ok=True)

                # 파일 저장
                filepath = DOWNLOAD_DIR / filename
                with open(filepath, 'wb') as f:
                    f.write(payload)

                print(f"다운로드 완료: {filename}")
                downloaded_files.append((filepath, filename, uid_text))

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

    # .env 로드 (GMAIL_USER / GMAIL_APP_PASSWORD)
    load_env_file()

    # 저장소 DB -> 로컬 캐시로 복사 (iCloud 동기화로 인한 disk I/O error 방지)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if DB_FILE.exists():
        shutil.copy2(DB_FILE, WORK_DB)
    elif WORK_DB.exists():
        WORK_DB.unlink()

    # DB 초기화
    conn = init_database()
    print(f"작업 DB: {WORK_DB}")
    print(f"최종 DB: {DB_FILE}")

    imap = None
    try:
        # Gmail IMAP 연결
        print("\nGmail IMAP 연결 중...")
        imap = get_imap_connection()
        print("Gmail IMAP 연결 성공!")

        # 메일 검색
        print(f"\n메일 검색 중... (제목: {MAIL_SUBJECT}, 이후: {SEARCH_SINCE})")
        uids = search_email_uids(imap)
        print(f"찾은 메일 수: {len(uids)}")

        # 이미 처리한 메일(UID)은 본문 fetch 없이 건너뜀.
        # 매 실행마다 전체 메일을 다시 받으면 느리고, 장시간 fetch 중
        # SSL 소켓이 끊겨 새 메일까지 못 받는 원인이 된다.
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT email_id FROM reports WHERE email_id IS NOT NULL')
        known_uids = {str(row[0]) for row in cursor.fetchall()}

        # 각 메일 처리
        total_downloaded = 0
        total_processed = 0

        for uid in uids:
            uid_text = uid.decode('ascii', errors='ignore') if isinstance(uid, bytes) else str(uid)
            if uid_text in known_uids:
                continue

            # 메일 1건 처리 실패가 전체를 중단시키지 않도록 개별 보호.
            # 소켓/SSL 오류는 연결 자체가 죽은 것이므로 재연결 후 1회 재시도한다.
            downloaded_files = []
            for attempt in (1, 2):
                try:
                    downloaded_files = process_message(imap, uid, conn)
                    break
                except (imaplib.IMAP4.abort, imaplib.IMAP4.error, OSError) as e:
                    print(f"메일 처리 오류 (uid={uid_text}, 시도 {attempt}): {e}")
                    if attempt == 1:
                        try:
                            imap.logout()
                        except Exception:
                            pass
                        try:
                            imap = get_imap_connection()
                            print("IMAP 재연결 성공")
                        except Exception as re_err:
                            print(f"IMAP 재연결 실패: {re_err}")
                            raise
                except Exception as e:
                    print(f"메일 처리 오류 (uid={uid_text}): {e}")
                    break

            for filepath, filename, message_id in downloaded_files:
                total_downloaded += 1
                if parse_excel_report(filepath, filename, message_id, conn):
                    total_processed += 1

        print("\n" + "=" * 60)
        print(f"완료!")
        print(f"다운로드: {total_downloaded}개")
        print(f"DB 저장: {total_processed}개")
        print("=" * 60)

    except RuntimeError as e:
        print(f"\n오류: {e}")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if imap is not None:
            try:
                imap.logout()
            except Exception:
                pass
        conn.close()
        # 로컬 캐시 DB -> 저장소 DB 로 복사 (Streamlit/git 이 읽는 위치)
        try:
            if WORK_DB.exists():
                shutil.copy2(WORK_DB, DB_FILE)
                print(f"DB 복사 완료: {WORK_DB} -> {DB_FILE}")
        except Exception as e:
            print(f"DB 복사 오류: {e}")


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
