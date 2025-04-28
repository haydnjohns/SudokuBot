#!/bin/bash

# File to send
SCRIPT_NAME="runonpi.py"

# Step 1: Copy the file to the Raspberry Pi
scp "/Users/haydnjohns/Documents/Coding/Python/SudokuBot/$SCRIPT_NAME" pi@raspberrypi.local:/home/pi/SudokuBot/

# Step 2: SSH into the Pi and run commands
ssh pi@raspberrypi.local << EOF
    cd ~/SudokuBot
    source venv/bin/activate
    python "$SCRIPT_NAME"
EOF
