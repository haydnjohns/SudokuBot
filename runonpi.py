import threading
from time import sleep
from gpiozero import OutputDevice

def initialise_steppers():
    left_stepper_pins = [OutputDevice(5), OutputDevice(6), OutputDevice(16), OutputDevice(20)]
    right_stepper_pins = [OutputDevice(14), OutputDevice(15), OutputDevice(23), OutputDevice(24)]
    increments_per_revolution = 4096  # your motor specs
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
    return left_stepper_pins, right_stepper_pins, step_sequence, increments_per_revolution

LEFT_STEPPER_PINS, RIGHT_STEPPER_PINS, STEP_SEQUENCE, INCREMENTS_PER_REVOLUTION = initialise_steppers()

def set_increment(pins, increment):
    for pin, value in zip(pins, increment):
        pin.value = value

def move_stepper(pins, revolutions, direction):
    total_increments = int(abs(revolutions) * INCREMENTS_PER_REVOLUTION)
    if total_increments == 0:
        return

    # Estimate reasonable delay based on some movement speed
    delay = 0.0015  # adjust depending on your motor's capability

    sequence = STEP_SEQUENCE if direction == "forward" else STEP_SEQUENCE[::-1]

    for increment in range(total_increments):
        increment_step = sequence[increment % len(sequence)]
        set_increment(pins, increment_step)
        sleep(delay)

def move_both_steppers(left_revolutions, right_revolutions, left_direction, right_direction):
    left_thread = threading.Thread(target=move_stepper,
                                   args=(LEFT_STEPPER_PINS, left_revolutions, left_direction))
    right_thread = threading.Thread(target=move_stepper,
                                    args=(RIGHT_STEPPER_PINS, right_revolutions, right_direction))

    left_thread.start()
    right_thread.start()

    left_thread.join()
    right_thread.join()

def control_steppers(move_sequence):
    for move in move_sequence:
        distance_mm, motor, direction, revolutions = move  # ← Correct unpacking!

        if motor == "left":
            move_stepper(LEFT_STEPPER_PINS, revolutions, direction)
        elif motor == "right":
            move_stepper(RIGHT_STEPPER_PINS, revolutions, direction)
        elif motor == "both":
            move_both_steppers(revolutions, revolutions, direction, direction)

# Example path
robot_path = [
    (0.24997130421432334, 'left', 'forward', 0.14515845335332597),
    (364.802382736357, 'both', 'forward', 3.1383839169833903),
    (0.38596877466992846, 'left', 'forward', 0.22413224809887947)
]

control_steppers(robot_path)