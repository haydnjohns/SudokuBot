#!/usr/bin/env python3
"""
Runs _on the Raspberry Pi_.  It listens on TCP :9999 for single-byte commands:
    w - forward    s - backward
    a - turn left  d - turn right
      space - de-energise coils
    q - shutdown server cleanly

The stepper driver code is the same as your previous version, just wrapped
inside a tiny socket server.
"""

import socket
import signal
import sys
import threading
from time import sleep

from gpiozero import OutputDevice

# ────────────────────────── Global State for Movement ──────────────────────────

movement_threads = [] # To keep track of active motor threads
stop_event = threading.Event() # To signal motor threads to stop
current_command = ' ' # Track the last movement command

# ──────────────────────────  Stepper helpers  ──────────────────────────


def initialise_steppers():
    left = [OutputDevice(i) for i in (5, 6, 16, 20)]
    right = [OutputDevice(i) for i in (14, 15, 23, 24)]
    sequence = [
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (0, 1, 0, 0),
        (0, 1, 1, 0),
        (0, 0, 1, 0),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
    ]
    return left, right, sequence, 4096  # 28BYJ-48 defaults


LEFT_PINS, RIGHT_PINS, STEP_SEQ, STEPS_PER_REV = initialise_steppers()


def set_increment(pins, inc):
    for p, v in zip(pins, inc):
        p.value = v


def _move_stepper_continuous(pins, stop_event_local, forwards=True, delay=0.0015):
    """Worker function for continuous movement in a thread."""
    seq = STEP_SEQ if forwards else STEP_SEQ[::-1]
    step_index = 0
    while not stop_event_local.is_set():
        set_increment(pins, seq[i % len(seq)])
        sleep(delay)
        step_index += 1
    # De-energize when stopped
    set_increment(pins, (0, 0, 0, 0))
    print(f"[Pi] Stepper thread stopped for pins: {[p.pin for p in pins]}")


def stop_movement():
    """Signals all active movement threads to stop and waits for them."""
    global movement_threads, stop_event, current_command
    if not movement_threads:
        # Ensure coils are de-energized even if no threads are running
        set_increment(LEFT_PINS, (0, 0, 0, 0))
        set_increment(RIGHT_PINS, (0, 0, 0, 0))
        return

    print("[Pi] Stopping movement...")
    stop_event.set() # Signal threads to stop
    for t in movement_threads:
        t.join(timeout=0.5) # Wait briefly for threads to finish
    movement_threads = []
    stop_event.clear() # Reset event for next movement
    # Explicitly de-energize after threads should have finished
    set_increment(LEFT_PINS, (0, 0, 0, 0))
    set_increment(RIGHT_PINS, (0, 0, 0, 0))
    print("[Pi] Movement stopped.")
    current_command = ' '


def start_movement(left_fwd=True, right_fwd=True, delay=0.0015):
    """Stops previous movement and starts new continuous movement."""
    global movement_threads, stop_event
    stop_movement() # Stop any existing movement first

    print(f"[Pi] Starting movement: left_fwd={left_fwd}, right_fwd={right_fwd}, delay={delay}")
    # Pass the global stop_event to each thread
    lt = threading.Thread(target=_move_stepper_continuous, args=(LEFT_PINS, stop_event, left_fwd, delay), daemon=True)
    rt = threading.Thread(target=_move_stepper_continuous, args=(RIGHT_PINS, stop_event, right_fwd, delay), daemon=True)
    movement_threads = [lt, rt]
    lt.start()
    rt.start()


# Define speeds
NORMAL_DELAY = 0.0015
SPRINT_DELAY = 0.0008 # Faster speed


def handle(cmd: str):
    global current_command
    if cmd == current_command and cmd != ' ': # Avoid restarting if command is the same
        return

    print(f"[Pi] Handling command: '{cmd}'")
    current_command = cmd # Update current command immediately

    if cmd == "w": start_movement(True, True, NORMAL_DELAY)         # forward
    elif cmd == "s": start_movement(False, False, NORMAL_DELAY)     # backward
    elif cmd == "a": start_movement(False, True, NORMAL_DELAY)      # turn left
    elif cmd == "d": start_movement(True, False, NORMAL_DELAY)      # turn right
    elif cmd == "W": start_movement(True, True, SPRINT_DELAY)       # SPRINT forward
    elif cmd == "S": start_movement(False, False, SPRINT_DELAY)     # SPRINT backward
    elif cmd == "A": start_movement(False, True, SPRINT_DELAY)      # SPRINT turn left
    elif cmd == "D": start_movement(True, False, SPRINT_DELAY)      # SPRINT turn right
    elif cmd == " ": stop_movement()                                # stop


# ─────────────────────────────  Server  ────────────────────────────────


def cleanup(*_):
    set_increment(LEFT_PINS, (0, 0, 0, 0))
    stop_movement() # Ensure motors are stopped and de-energized
    print("[Pi] Cleanup complete. Exiting.")
    sys.exit(0) # Ensure the script exits


for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, cleanup)


def run_server(host: str = "0.0.0.0", port: int = 9999):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        print(f"[Pi] Listening on {host}:{port}")
    except OSError as e:
        print(f"[Pi] Error binding to {host}:{port} - {e}", file=sys.stderr)
        sys.exit(1) # Exit if binding fails

    # Use the created socket 's'
    with s:
        conn, addr = s.accept()
        print(f"[Pi] Client {addr} connected")
        with conn:
            while True:
                try:
                    data = conn.recv(1)
                    if not data:
                        print("[Pi] Client disconnected (received empty data)")
                        break
                    cmd = data.decode("utf-8")
                    # print(f"[Pi] Received command: '{cmd}'") # Logging moved to handle()
                    if cmd == "q":
                        print("[Pi] Quit command received. Shutting down.")
                        break
                    handle(cmd)
                except ConnectionResetError:
                    print("[Pi] Client connection reset.")
                    break
                except Exception as e:
                    print(f"[Pi] Error handling client data: {e}", file=sys.stderr)
                    break

    print("[Pi] Server loop finished.")
    cleanup()


if __name__ == "__main__":
    run_server()
