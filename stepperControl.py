from gpiozero import OutputDevice
from time import sleep
import threading  # Allows motors to run simultaneously

# Define GPIO pins for Stepper 1
STEPPER_1_PINS = [OutputDevice(5), OutputDevice(6), OutputDevice(16), OutputDevice(20)]

# Define GPIO pins for Stepper 2
STEPPER_2_PINS = [OutputDevice(21), OutputDevice(22), OutputDevice(23), OutputDevice(24)]

# Step sequence for half-stepping
STEP_SEQUENCE = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1]
]

def set_step(pins, values):
    """ Set motor pins based on the step sequence """
    for pin, value in zip(pins, values):
        pin.value = value

def move_stepper(motor, direction, steps, delay=0.002):
    """ Moves the selected stepper motor in the specified direction for a given number of steps. """
    sequence = STEP_SEQUENCE[::-1] if direction == 'b' else STEP_SEQUENCE
    for _ in range(steps):
        for step in sequence:
            set_step(motor, step)
            sleep(delay)
    set_step(motor, [0, 0, 0, 0])  # Turn off motor after movement

def run_motor(motor_id, direction, steps):
    """ Runs a motor in a separate thread """
    if motor_id == 1:
        move_stepper(STEPPER_1_PINS, direction, steps)
    elif motor_id == 2:
        move_stepper(STEPPER_2_PINS, direction, steps)

try:
    while True:
        motor_input = input("Enter motor (1 or 2, or 'both' for both motors), or 'q' to quit: ").strip().lower()
        if motor_input == 'q':
            break
        elif motor_input == '1':
            motors = [(1, STEPPER_1_PINS)]
        elif motor_input == '2':
            motors = [(2, STEPPER_2_PINS)]
        elif motor_input == 'both':
            motors = [(1, STEPPER_1_PINS), (2, STEPPER_2_PINS)]
        else:
            print("Invalid input. Enter 1, 2, or 'both'.")
            continue

        direction = input("Enter direction ('f' for forward, 'b' for backward): ").strip().lower()
        if direction not in ['f', 'b']:
            print("Invalid direction. Use 'f' or 'b'.")
            continue

        try:
            steps = int(input("Enter number of steps: "))
        except ValueError:
            print("Invalid step count. Enter a number.")
            continue

        threads = []
        for motor_id, motor in motors:
            thread = threading.Thread(target=run_motor, args=(motor_id, direction, steps))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()  # Ensure all motors finish before asking for new input

except KeyboardInterrupt:
    print("\nStopping motors.")

finally:
    set_step(STEPPER_1_PINS, [0, 0, 0, 0])
    set_step(STEPPER_2_PINS, [0, 0, 0, 0])