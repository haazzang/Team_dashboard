#!/usr/bin/env python3
"""
Swap Report 자동 업데이트 스크립트

이 스크립트는:
1. Gmail에서 새 Swap Report를 가져옴
2. DB를 업데이트
3. 변경사항이 있으면 git commit & push

사용법:
    python automation/swap/auto_update_swap_reports.py

cron 설정 (매일 오후 7시):
    0 19 * * * cd /Users/hyejinha/Desktop/Workspace/Team && /usr/bin/python3 automation/swap/auto_update_swap_reports.py >> auto_update.log 2>&1

launchd 설정 (Mac):
    아래 plist 파일 참조
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
LOG_FILE = ROOT_DIR / 'auto_update.log'
FETCHER_PATH = SCRIPT_DIR / "swap_report_fetcher.py"


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
            cwd=cwd or ROOT_DIR,
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
    success, stdout, stderr = run_command(f'{sys.executable} "{FETCHER_PATH}"', cwd=ROOT_DIR)

    if not success:
        log(f"오류: {stderr}")
        return False

    log(stdout)

    # 2. 배포 브랜치(master)의 DB 갱신
    #    main/master 가 갈라져 있어 `git push main:master` 는 거부되므로,
    #    git plumbing 으로 origin/master 위에 swap_reports.db 한 파일만 바꾼
    #    커밋을 만들어 push 한다. 작업 트리/로컬 브랜치를 전혀 건드리지 않고
    #    항상 fast-forward 가 보장된다.
    commit_msg = f"Auto-update swap_reports.db ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
    if not push_db_to_master(commit_msg):
        return False

    log("=" * 50)
    log("자동 업데이트 완료!")
    log("=" * 50)

    return True


def push_db_to_master(commit_msg, deploy_branch='master', db_path='swap_reports.db'):
    """origin/<deploy_branch> 위에 swap_reports.db 만 교체한 커밋을 만들어 push.

    main/master 가 갈라져 있어도 항상 origin/master 를 부모로 삼으므로
    fast-forward push 가 보장된다.
    """
    db_file = ROOT_DIR / db_path
    if not db_file.exists():
        log(f"오류: {db_file} 없음")
        return False

    # 최신 origin/master 가져오기
    log(f"origin/{deploy_branch} fetch 중...")
    ok, out, err = run_command(f'git fetch origin {deploy_branch}', cwd=ROOT_DIR)
    if not ok:
        log(f"fetch 오류: {err}")
        return False

    # 현재 master 의 db blob 과 비교 -> 동일하면 스킵
    ok, master_blob, _ = run_command(
        f'git rev-parse origin/{deploy_branch}:{db_path}', cwd=ROOT_DIR
    )
    # -w 필수: 해시만 계산하면 blob 이 object db 에 없어 write-tree 가
    # "invalid object" 로 실패한다.
    ok2, new_blob, err2 = run_command(f'git hash-object -w "{db_file}"', cwd=ROOT_DIR)
    if not ok2:
        log(f"hash-object 오류: {err2}")
        return False
    if ok and master_blob.strip() == new_blob.strip():
        log(f"{deploy_branch} 의 DB 가 이미 최신 - 업데이트 불필요")
        return True

    # 임시 인덱스에 origin/master 트리를 읽고 db 만 교체 -> 새 트리/커밋 생성
    tmp_index = ROOT_DIR / '.git' / 'tmp-master-index'
    env_prefix = f'GIT_INDEX_FILE="{tmp_index}" '
    try:
        steps = [
            f'{env_prefix}git read-tree origin/{deploy_branch}',
            f'{env_prefix}git update-index --add --cacheinfo 100644,{new_blob.strip()},{db_path}',
        ]
        for cmd in steps:
            ok, out, err = run_command(cmd, cwd=ROOT_DIR)
            if not ok:
                log(f"인덱스 구성 오류: {cmd}\n{err}")
                return False

        ok, tree, err = run_command(f'{env_prefix}git write-tree', cwd=ROOT_DIR)
        if not ok:
            log(f"write-tree 오류: {err}")
            return False
        tree = tree.strip()

        ok, commit, err = run_command(
            f'git commit-tree {tree} -p origin/{deploy_branch} -m "{commit_msg}"',
            cwd=ROOT_DIR,
        )
        if not ok:
            log(f"commit-tree 오류: {err}")
            return False
        commit = commit.strip()

        log(f"{deploy_branch} 에 push 중... ({commit[:8]})")
        ok, out, err = run_command(
            f'git push origin {commit}:refs/heads/{deploy_branch}', cwd=ROOT_DIR
        )
        if not ok:
            log(f"push 오류: {err}")
            return False
        log(f"Push 완료! ({deploy_branch} <- {commit[:8]}): {commit_msg}")
        return True
    finally:
        if tmp_index.exists():
            tmp_index.unlink()


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
