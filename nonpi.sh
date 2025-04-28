#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  runonpi.sh – deploy the stepper server to the Raspberry Pi and start the
#               local WASD client in one go.
# ---------------------------------------------------------------------------
set -euo pipefail

REMOTE_HOST="pi@raspberrypi.local"
REMOTE_DIR="/home/pi/SudokuBot"
REMOTE_SCRIPT="_stepper_server.py"   # <-- name created by the patcher
LOCAL_CLIENT="sd_control.py"         # <-- name created by the patcher

# 1. Copy the server code to the Pi -------------------------------------------------
echo "[1/3] Copying ${REMOTE_SCRIPT} to ${REMOTE_HOST}:${REMOTE_DIR}"
scp "${REMOTE_SCRIPT}" "${REMOTE_HOST}:${REMOTE_DIR}/"

# 2. Restart the server on the Pi (kill any previous instance) ----------------------
echo "[2/3] (Re)starting remote server …"
ssh "${REMOTE_HOST}" <<'SSH_CMDS'
  set -e
  cd ~/SudokuBot
  source venv/bin/activate
  pkill -f _stepper_server.py 2>/dev/null || true
  nohup python3 _stepper_server.py > pi_stepper_server.log 2>&1 &
  echo "[Pi] _stepper_server.py running   (log: ~/SudokuBot/pi_stepper_server.log)"
SSH_CMDS

# 3. Launch the local WASD client ---------------------------------------------------
echo "[3/3] Launching local client – press 'q' to quit."
python3 "${LOCAL_CLIENT}" --host raspberrypi.local

echo "Done."
