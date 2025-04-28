#!/usr/bin/env bash
set -euo pipefail

# ─── User-adjustable bits ──────────────────────────────────────────────────────
PI_HOST="pi@raspberrypi.local"          # the host you ssh into
PI_PROJ_DIR="/home/pi/SudokuBot"        # folder on the Pi
REMOTE_SCRIPT="robot_server.py"         # name we use on both sides
# ───────────────────────────────────────────────────────────────────────────────

echo "Uploading latest server code…"
scp "${REMOTE_SCRIPT}" "${PI_HOST}:${PI_PROJ_DIR}/"

echo "Stopping any existing server on the Pi…"
ssh "${PI_HOST}" "pkill -f ${REMOTE_SCRIPT} || true"

echo "Starting server inside virtual-env…"
ssh "${PI_HOST}" <<'SSHCMDS'
  set -e
  cd ~/SudokuBot
  source venv/bin/activate
  nohup python robot_server.py > robot_server.log 2>&1 &
SSHCMDS

echo "✓ Server running on Pi."
echo "Now run:  python control_robot.py raspberrypi.local"
