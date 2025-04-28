from gpiozero import OutputDevice
from time import sleep

# Define GPIO pins for left stepper motor (adjust according to your setup)
left_stepper_pins = [OutputDevice(5), OutputDevice(6), OutputDevice(16), OutputDevice(20)]

# Define GPIO pins for right stepper motor (adjust according to your setup)
right_stepper_pins = [OutputDevice(15), OutputDevice(16), OutputDevice(23), OutputDevice(24)]

# Step sequence for half-stepping (8 steps for half-step)
step_sequence = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1]
]

# Set stepper motor to the current step
def set_increment(pins, increment):
    for pin, value in zip(pins, increment):
        pin.value = value

# Move stepper motor forward
def move_stepper(pins, steps, delay):
    for i in range(steps):
        step = step_sequence[i % len(step_sequence)]  # Cycle through the step sequence
        set_increment(pins, step)
        sleep(delay)

# Move both motors forward by 1/4 turn (128 steps for each motor)
steps_per_revolution = 512
steps_for_quarter_turn = steps_per_revolution // 4
delay_per_step = 0.02  # Adjust speed by changing the delay

# Move both motors
move_stepper(left_stepper_pins, steps_for_quarter_turn, delay_per_step)
move_stepper(right_stepper_pins, steps_for_quarter_turn, delay_per_step)