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
  set -e # Exit immediately if a command exits with a non-zero status.
  cd ~/SudokuBot
  # Ensure the virtual environment exists and is activated
  if [ ! -d "venv" ]; then
    echo "[Pi] Error: Virtual environment 'venv' not found in ~/SudokuBot."
    echo "[Pi] Please create it and install dependencies (e.g., gpiozero)."
    exit 1
  fi
  source venv/bin/activate

  echo "[Pi] Stopping previous server (if any)..."
  pkill -f _stepper_server.py || true # Allow command to fail if no process exists
  sleep 1 # Give time for process to terminate cleanly

  echo "[Pi] Starting new server in background..."
  nohup python3 _stepper_server.py > pi_stepper_server.log 2>&1 &
  SERVER_PID=$!
  sleep 2 # Give the server a moment to start up or potentially fail

  # Check if the process is still running and listening on the port
  if ps -p $SERVER_PID > /dev/null && ss -tuln | grep -q ':9999'; then
    echo "[Pi] Server process $SERVER_PID seems to be running and listening on port 9999."
    echo "[Pi] Log file: ~/SudokuBot/pi_stepper_server.log"
  else
    echo "[Pi] Error: Server process failed to start or stay running."
    echo "[Pi] Check the log file for details: ~/SudokuBot/pi_stepper_server.log"
    # Optional: Exit SSH session with an error if the server failed to start
    # exit 1
  fi
SSH_CMDS

# Optional: Check the exit status of the SSH command if 'exit 1' was added above
# SSH_EXIT_STATUS=$?
# if [ $SSH_EXIT_STATUS -ne 0 ]; then
#   echo "[Error] SSH command indicated server startup failure on the Pi."
#   echo "        Please check the log: ssh ${REMOTE_HOST} 'cat ~/SudokuBot/pi_stepper_server.log'"
#   exit $SSH_EXIT_STATUS
# fi

# 3. Launch the local WASD client ---------------------------------------------------
echo "[3/3] Launching local client – press 'q' to quit."
python3 "${LOCAL_CLIENT}" --host raspberrypi.local

echo "Done."
