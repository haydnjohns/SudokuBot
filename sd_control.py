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

    print("Controls:  w/a/s/d – drive | space – stop | q – quit")
    try:
        while True:
            key = read_key()
            if key is None:
                continue

            # Normalise to lower-case and filter allowed keys
            key = key.lower()
            if key in ("w", "a", "s", "d", " ", "q"):
                sock.send(key.encode("utf-8"))
                if key == "q":
                    break
    finally:
        sock.close()
        print("\n[Client] Disconnected")


if __name__ == "__main__":
    main()
