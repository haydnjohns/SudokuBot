#!/usr/bin/env python3
"""
Tiny client that lives on your _local workstation_.
Grabs single-key presses and streams them to the Raspberry Pi.
"""

import argparse
import select
import socket
import sys
import termios
import tty


def read_key(timeout: float = 0.1):
    """Non-blocking single keystroke reader (POSIX only)."""
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([fd], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
    return None


def main():
    p = argparse.ArgumentParser(description="WASD tele-op client")
    p.add_argument("--host", default="raspberrypi.local", help="Pi hostname / IP")
    p.add_argument("--port", type=int, default=9999)
    args = p.parse_args()

    print(f"[Client] Connecting to {args.host}:{args.port} …")
    try:
        sock = socket.create_connection((args.host, args.port), timeout=5)
    except OSError as e:
        sys.exit(f"[Client] Connection error: {e}")

    print("Controls:  w/a/s/d – drive | W/A/S/D - SPRINT | space – stop | q – quit")
    last_sent_key = ' ' # Track the last key sent, start with stop state
    try:
        while True:
            key = read_key()

            if key is None:
                # Key released
                if last_sent_key in ("w", "a", "s", "d", "W", "A", "S", "D"):
                    print("[Client] Key released, sending stop.")
                    sock.send(" ".encode("utf-8"))
                    last_sent_key = ' '
                continue

            # Key pressed
            # No lower() here, case matters for sprint
            allowed_keys = ("w", "a", "s", "d", "W", "A", "S", "D", " ", "q")

            if key in allowed_keys:
                if key == "q":
                    sock.send(key.encode("utf-8"))
                    break
                if key != last_sent_key:
                    print(f"[Client] Sending command: '{key}'")
                    sock.send(key.encode("utf-8"))
                    last_sent_key = key
            # Ignore other keys silently

    except Exception as e:
        print(f"\n[Client] Error: {e}")
    finally:
        # Ensure stop command is sent on exit if moving
        if last_sent_key != ' ':
            try:
                sock.send(key.encode("utf-8"))
            except Exception as e_final:
                print(f"[Client] Error sending final stop: {e_final}")
        sock.close()
        print("\n[Client] Disconnected")


if __name__ == "__main__":
    main()
