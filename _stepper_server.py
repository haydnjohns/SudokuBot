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

DEFAULT_DELAY = 0.0015
SPRINT_DELAY = 0.0008 # Faster! Tune this value to taste/motor capability

def set_increment(pins, inc):
    for p, v in zip(pins, inc):
        p.value = v


def move_stepper(pins, steps, forwards=True, delay=DEFAULT_DELAY):
    seq = STEP_SEQ if forwards else STEP_SEQ[::-1]
    for i in range(steps):
        set_increment(pins, seq[i % len(seq)])
        sleep(delay)
    # Keep coils energized between commands for holding torque? Optional.
    # set_increment(pins, (0, 0, 0, 0)) # release coils


def move(left_steps, right_steps, left_fwd=True, right_fwd=True):
    lt = threading.Thread(target=move_stepper, args=(LEFT_PINS, left_steps, left_fwd))
    rt = threading.Thread(target=move_stepper, args=(RIGHT_PINS, right_steps, right_fwd))
    lt.start(), rt.start()
    lt.join(), rt.join()


# Roughly 0.05 rev per key-press → tune to taste
STEP_SIZE = int(STEPS_PER_REV * 0.05)


def handle(cmd: str):
    if cmd == "w":        # forward
        move(STEP_SIZE, STEP_SIZE, True, True, delay=DEFAULT_DELAY)
    elif cmd == "s":      # backward
        move(STEP_SIZE, STEP_SIZE, False, False, delay=DEFAULT_DELAY)
    elif cmd == "a":      # turn left  (left backwards, right forwards)
        move(STEP_SIZE, STEP_SIZE, False, True, delay=DEFAULT_DELAY)
    elif cmd == "d":      # turn right (left forwards, right backwards)
        move(STEP_SIZE, STEP_SIZE, True, False, delay=DEFAULT_DELAY)
    elif cmd == "e":      # sprint forward
        move(STEP_SIZE, STEP_SIZE, True, True, delay=SPRINT_DELAY)
    elif cmd == " ":      # stop (just de-energise)
        set_increment(LEFT_PINS, (0, 0, 0, 0))
        set_increment(RIGHT_PINS, (0, 0, 0, 0))


# ─────────────────────────────  Server  ────────────────────────────────


def cleanup(*_):
    set_increment(LEFT_PINS, (0, 0, 0, 0))
    set_increment(RIGHT_PINS, (0, 0, 0, 0))
    sys.exit(0)


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
                    print(f"[Pi] Received command: '{cmd}'") # Add logging
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
