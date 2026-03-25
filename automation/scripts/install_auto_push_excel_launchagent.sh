#!/usr/bin/env bash
set -euo pipefail

LABEL="com.hyejinha.team-dashboard.excel-autopush"
REPO_ROOT="/Users/hyejinha/Desktop/Workspace/Team"
SOURCE_SCRIPT="${REPO_ROOT}/automation/scripts/auto_push_excel.sh"
SOURCE_PLIST="${REPO_ROOT}/automation/launchd/${LABEL}.plist"
APP_SUPPORT_DIR="${HOME}/Library/Application Support/TeamDashboard"
INSTALLED_SCRIPT="${APP_SUPPORT_DIR}/auto_push_excel.sh"
TARGET_PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"

mkdir -p "${HOME}/Library/LaunchAgents" "${HOME}/Library/Logs" "${APP_SUPPORT_DIR}"
install -m 755 "${SOURCE_SCRIPT}" "${INSTALLED_SCRIPT}"
rm -f "${TARGET_PLIST}"
cp "${SOURCE_PLIST}" "${TARGET_PLIST}"

launchctl bootout "gui/$(id -u)" "${TARGET_PLIST}" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "${TARGET_PLIST}"
launchctl kickstart -k "gui/$(id -u)/${LABEL}"

echo "Installed and started ${LABEL}"
