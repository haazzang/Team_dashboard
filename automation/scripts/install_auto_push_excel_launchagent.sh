#!/usr/bin/env bash
set -euo pipefail

LABEL="com.hyejinha.team-dashboard.excel-autopush"
REPO_ROOT="/Users/hyejinha/Desktop/Workspace/Team"
SOURCE_SCRIPT="${REPO_ROOT}/automation/scripts/auto_push_excel.sh"
SOURCE_PLIST="${REPO_ROOT}/automation/launchd/${LABEL}.plist"
INSTALL_BIN_DIR="${HOME}/.local/bin"
INSTALLED_SCRIPT="${INSTALL_BIN_DIR}/team-dashboard-auto-push-excel.sh"
TARGET_PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"

mkdir -p "${HOME}/Library/LaunchAgents" "${HOME}/Library/Logs" "${INSTALL_BIN_DIR}"
install -m 755 "${SOURCE_SCRIPT}" "${INSTALLED_SCRIPT}"
rm -f "${TARGET_PLIST}"
cp "${SOURCE_PLIST}" "${TARGET_PLIST}"

launchctl bootout "gui/$(id -u)" "${TARGET_PLIST}" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "${TARGET_PLIST}"
launchctl kickstart -k "gui/$(id -u)/${LABEL}"

echo "Installed and started ${LABEL}"
