#!/usr/bin/env bash
set -euo pipefail

FILE_NAME="${FILE_NAME:-2026_멀티.xlsx}"
REMOTE="${REMOTE:-origin}"
MAIN_BRANCH="${MAIN_BRANCH:-main}"
MASTER_BRANCH="${MASTER_BRANCH:-master}"

if ! command -v fswatch >/dev/null 2>&1; then
  echo "fswatch not found. Install with: brew install fswatch"
  exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
FILE_PATH="${REPO_ROOT}/${FILE_NAME}"
WATCH_DIR=$(dirname "$FILE_PATH")

if [[ ! -f "$FILE_PATH" ]]; then
  echo "File not found: $FILE_PATH"
  exit 1
fi

current_branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" != "$MAIN_BRANCH" ]]; then
  echo "Please run this on branch '$MAIN_BRANCH'. Current: $current_branch"
  exit 1
fi

push_to_master() {
  local commit="$1"
  local tmp_dir
  tmp_dir=$(mktemp -d)

  git -C "$REPO_ROOT" fetch "$REMOTE" "$MASTER_BRANCH"
  git -C "$REPO_ROOT" worktree add "$tmp_dir" "$REMOTE/$MASTER_BRANCH"

  cleanup() {
    set +e
    git -C "$REPO_ROOT" worktree remove --force "$tmp_dir" >/dev/null 2>&1
    rm -rf "$tmp_dir"
  }
  trap cleanup RETURN

  git -C "$tmp_dir" cherry-pick "$commit"
  git -C "$tmp_dir" push "$REMOTE" HEAD:"$MASTER_BRANCH"
}

get_mtime() {
  if stat -f %m "$1" >/dev/null 2>&1; then
    stat -f %m "$1"
  else
    stat -c %Y "$1"
  fi
}

last_mtime=""

echo "Watching: $FILE_PATH"
fswatch -o "$WATCH_DIR" | while read -r _; do
  sleep 2
  if [[ ! -f "$FILE_PATH" ]]; then
    continue
  fi

  mtime=$(get_mtime "$FILE_PATH")
  if [[ -n "$last_mtime" && "$mtime" == "$last_mtime" ]]; then
    continue
  fi
  last_mtime="$mtime"

  git -C "$REPO_ROOT" add "$FILE_NAME"
  if git -C "$REPO_ROOT" diff --cached --quiet -- "$FILE_NAME"; then
    continue
  fi

  ts=$(date "+%Y-%m-%d %H:%M:%S")
  git -C "$REPO_ROOT" commit -m "Update $FILE_NAME ($ts)" -- "$FILE_NAME"
  commit=$(git -C "$REPO_ROOT" rev-parse HEAD)

  git -C "$REPO_ROOT" push "$REMOTE" "$MAIN_BRANCH"
  push_to_master "$commit"
done
