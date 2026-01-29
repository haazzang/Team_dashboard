#!/usr/bin/env python3
"""
Swap Report 자동 업데이트 스크립트

이 스크립트는:
1. Gmail에서 새 Swap Report를 가져옴
2. DB를 업데이트
3. 변경사항이 있으면 git commit & push

사용법:
    python auto_update_swap_reports.py

cron 설정 (매일 오후 7시):
    0 19 * * * cd /Users/hyejinha/Desktop/Workspace/Team && /usr/bin/python3 auto_update_swap_reports.py >> auto_update.log 2>&1

launchd 설정 (Mac):
    아래 plist 파일 참조
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 스크립트 디렉토리
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_FILE = SCRIPT_DIR / 'auto_update.log'


def log(message):
    """로그 출력"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg + '\n')


def run_command(cmd, cwd=None):
    """명령어 실행"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd or SCRIPT_DIR,
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, '', str(e)


def main():
    log("=" * 50)
    log("Swap Report 자동 업데이트 시작")
    log("=" * 50)

    # 1. swap_report_fetcher.py 실행
    log("Gmail에서 새 리포트 확인 중...")
    success, stdout, stderr = run_command(f'{sys.executable} swap_report_fetcher.py')

    if not success:
        log(f"오류: {stderr}")
        return False

    log(stdout)

    # 2. DB 변경사항 확인
    log("DB 변경사항 확인 중...")
    success, stdout, stderr = run_command('git status --porcelain swap_reports.db')

    if not stdout.strip():
        log("변경사항 없음 - 업데이트 불필요")
        return True

    log(f"변경 감지: {stdout.strip()}")

    # 3. git add
    log("git add 실행 중...")
    success, stdout, stderr = run_command('git add swap_reports.db')
    if not success:
        log(f"git add 오류: {stderr}")
        return False

    # 4. git commit
    log("git commit 실행 중...")
    commit_msg = f"Auto-update swap_reports.db ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
    success, stdout, stderr = run_command(f'git commit -m "{commit_msg}"')
    if not success:
        log(f"git commit 오류: {stderr}")
        return False

    log(f"커밋 완료: {commit_msg}")

    # 5. git push
    log("git push 실행 중...")
    success, stdout, stderr = run_command('git push origin main')
    if not success:
        log(f"git push 오류: {stderr}")
        return False

    log("Push 완료!")

    # master 브랜치도 업데이트
    success, stdout, stderr = run_command('git push origin main:master')
    if success:
        log("master 브랜치도 업데이트 완료")

    log("=" * 50)
    log("자동 업데이트 완료!")
    log("=" * 50)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
