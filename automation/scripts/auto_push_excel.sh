#!/usr/bin/env bash
set -euo pipefail

FILE_NAME="${FILE_NAME:-2026_멀티.xlsx}"
REMOTE="${REMOTE:-origin}"
MAIN_BRANCH="${MAIN_BRANCH:-main}"
MASTER_BRANCH="${MASTER_BRANCH:-master}"
ENABLE_MASTER_SYNC="${ENABLE_MASTER_SYNC:-1}"

if ! command -v fswatch >/dev/null 2>&1; then
  echo "fswatch not found. Install with: brew install fswatch"
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-}"
if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT=$(git rev-parse --show-toplevel)
fi
REPO_ROOT=$(cd "$REPO_ROOT" && pwd)
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

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

push_to_master() {
  local commit="$1"
  local commit_subject="$2"
  local tmp_dir=""

  if [[ "$ENABLE_MASTER_SYNC" != "1" ]]; then
    log "Master sync disabled. Skipping '$MASTER_BRANCH'."
    return 0
  fi

  if ! git -C "$REPO_ROOT" ls-remote --exit-code --heads "$REMOTE" "$MASTER_BRANCH" >/dev/null 2>&1; then
    log "Remote branch '$MASTER_BRANCH' not found. Skipping."
    return 0
  fi

  tmp_dir=$(mktemp -d)

  git -C "$REPO_ROOT" fetch "$REMOTE" "$MASTER_BRANCH"
  git -C "$REPO_ROOT" worktree add --detach "$tmp_dir" "$REMOTE/$MASTER_BRANCH"

  cleanup() {
    local target="${tmp_dir:-}"
    if [[ -z "$target" ]]; then
      return 0
    fi
    set +e
    git -C "$REPO_ROOT" worktree remove --force "$target" >/dev/null 2>&1
    rm -rf "$target"
  }
  trap cleanup RETURN

  git -C "$tmp_dir" checkout "$commit" -- "$FILE_NAME"
  git -C "$tmp_dir" add -- "$FILE_NAME"
  if git -C "$tmp_dir" diff --cached --quiet -- "$FILE_NAME"; then
    log "No '$FILE_NAME' change needed on '$MASTER_BRANCH'."
    return 0
  fi

  git -C "$tmp_dir" commit -m "$commit_subject" -- "$FILE_NAME"
  git -C "$tmp_dir" push "$REMOTE" HEAD:"$MASTER_BRANCH"
  log "Synced '$FILE_NAME' to '$MASTER_BRANCH'."
}

get_signature() {
  if stat -f %m:%z "$1" >/dev/null 2>&1; then
    stat -f %m:%z "$1"
  else
    stat -c %Y:%s "$1"
  fi
}

sync_file() {
  if [[ ! -f "$FILE_PATH" ]]; then
    log "File not found: $FILE_PATH"
    return 0
  fi

  git -C "$REPO_ROOT" add -- "$FILE_NAME"
  if git -C "$REPO_ROOT" diff --cached --quiet -- "$FILE_NAME"; then
    return 0
  fi

  local ts commit commit_subject
  ts=$(date "+%Y-%m-%d %H:%M:%S")
  commit_subject="Update $FILE_NAME ($ts)"
  git -C "$REPO_ROOT" commit -m "$commit_subject" -- "$FILE_NAME"
  commit=$(git -C "$REPO_ROOT" rev-parse HEAD)
  log "Created commit $commit"

  git -C "$REPO_ROOT" push "$REMOTE" "$MAIN_BRANCH"
  log "Pushed '$FILE_NAME' to '$REMOTE/$MAIN_BRANCH'."

  push_to_master "$commit" "$commit_subject"
}

last_signature=""

log "Watching: $FILE_PATH"
sync_file
last_signature=$(get_signature "$FILE_PATH")

fswatch -o "$WATCH_DIR" | while read -r _; do
  sleep 2
  if [[ ! -f "$FILE_PATH" ]]; then
    continue
  fi

  signature=$(get_signature "$FILE_PATH")
  if [[ -n "$last_signature" && "$signature" == "$last_signature" ]]; then
    continue
  fi
  last_signature="$signature"
  sync_file
done
