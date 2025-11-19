import os
import numpy as np
import pygame
from affectively.environments.base import compute_confidence_interval
from affectively.environments.heist_game_obs import HeistEnvironmentGameObs
from affectively.utils.logging import InteractiveDashboard

# Headless mode (keep this if needed)
os.environ["SDL_VIDEODRIVER"] = "dummy"

# --- Global state ---
# Action space: [mouseX_bin, mouseY_bin, forward, sideways, shoot]
current_action = [4, 4, 0, 0, 0]  # Start at center (bin 4 = 0 for both mouse axes)

# --- Deadzone Configuration ---
DEADZONE = 0.25  # Threshold for deadzone (adjustable)

# Mouse bins from the Unity code
MOUSE_X_BINS = [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]
MOUSE_Y_BINS = [-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2]

# --- Initialize pygame and gamepad ---
pygame.init()

# Check if there are any gamepads connected
if pygame.joystick.get_count() == 0:
    print("No gamepad found. Gamepad required for FPS controls.")
    joystick = None
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Gamepad initialized:", joystick.get_name())

KEYBOARD_ENABLED = ("dummy" not in os.environ.get("SDL_VIDEODRIVER", ""))


def map_axis_to_bin(axis_value, bins, deadzone=0.15):
    """Map analog stick value (-1 to 1) to discrete bin index.
    Always uses the smallest sensitivity bins (closest to center)."""
    # Apply deadzone
    if abs(axis_value) < deadzone:
        # Return center bin (the one with value closest to 0)
        center_idx = len(bins) // 2
        return center_idx
    
    # Use the smallest non-zero sensitivity
    # For negative movement, use the bin just left of center
    # For positive movement, use the bin just right of center
    center_idx = len(bins) // 2
    
    if axis_value < 0:
        # Use bin to the left of center (smallest negative sensitivity)
        return center_idx - 1
    else:
        # Use bin to the right of center (smallest positive sensitivity)
        return center_idx + 1


def map_movement_axis(axis_value, deadzone=0.15):
    """Map analog stick to discrete movement: 0 (back/left), 1 (neutral), 2 (forward/right)."""
    if abs(axis_value) < deadzone:
        return 1  # Neutral
    elif axis_value < 0:  # Negative = backwards/left
        return 0
    else:  # Positive = forwards/right
        return 2


# --- Main simulation ---
if __name__ == "__main__":

    env = HeistEnvironmentGameObs(
        0,
        graphics=True,
        weight=0,
        discretize=False,
        cluster=3,
        decision_period=1,
        target_arousal=1,
        period_ra=False,
        classifier=False,
        preference=False,
        capture_fps=-60,
        sensitivity=0.5
    )

    env.callback = InteractiveDashboard(env)
    # state = env.reset()

    trigger_was_pressed = False  # Track trigger state for semi-auto shooting

    pause = True
    end = True
    env.episode_length = 6000

    while True:
        
        if pygame.joystick.get_count() == 0:
            print("No gamepad found.")
            try:
                joystick = pygame.joystick.Joystick(0)
                joystick.init()
                print("Gamepad initialized:", joystick.get_name())
            except:
                continue

        if env.customSideChannel.interactiveReset:
            end = True
            pause = True
            env.customSideChannel.interactiveReset = False

        if env.episode_length == 6000 or end or pause:

            if end or env.episode_length == 6000:
                env.callback.on_episode_end()
                pause = True
        
            for event in pygame.event.get():

                if event.type == pygame.JOYBUTTONUP:
                    if event.button == 6:

                        if env.episode_length == 6000 or end:
                            state = env.reset()
                            env.callback.clear()
                            end = False

                        env.callback._waiting_restart = not pause
                        env.callback.pause = not pause
                        pause = not pause
                        print("Episode restarted via gamepad.")

            continue


        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONUP:
                if event.button == 6:
                    pause = not pause
                    print(f"Pause toggled to {pause}")
                    if pause:
                        env.callback.on_pause()
                    continue


        if joystick is not None:

            # Left stick: Movement (forward/back and strafe left/right)
            left_x = joystick.get_axis(0)  # Strafe (sideways)
            left_y = joystick.get_axis(1)  # Forward/back

            # Right stick: Aiming (mouse look)
            right_x = joystick.get_axis(2)  # Look left/right (mouseX)
            right_y = joystick.get_axis(3)  # Look up/down (mouseY)

            mouseX_bin = map_axis_to_bin(right_x, MOUSE_X_BINS, DEADZONE)
            mouseY_bin = map_axis_to_bin(-right_y, MOUSE_Y_BINS, DEADZONE)  # Inverted for proper up/down
            
            current_action[0] = mouseX_bin
            current_action[1] = mouseY_bin

            forward = map_movement_axis(-left_y, DEADZONE)  # Inverted: push up = forward = 1
            sideways = map_movement_axis(left_x, DEADZONE)  # Right = 1
            
            current_action[2] = forward
            current_action[3] = sideways

            trigger_pressed_now = False
            
            try:
                if joystick.get_button(5):  # RB
                    trigger_pressed_now = True
                if joystick.get_button(7):  # RT as button
                    trigger_pressed_now = True
            except:
                pass
            
            # Check RT as analog trigger (axis 5)
            try:
                rt_axis = joystick.get_axis(5)
                # RT can be -1 to 1 (some controllers) or 0 to 1
                if rt_axis > 0.5:  # Trigger pressed
                    trigger_pressed_now = True
            except:
                pass

            # Only shoot on the rising edge (press, not hold)
            if trigger_pressed_now and not trigger_was_pressed:
                current_action[4] = 1  # Fire!
            else:
                current_action[4] = 0  # Don't fire
            
            # Update trigger state for next frame
            trigger_was_pressed = trigger_pressed_now

        state, reward, terminated, info = env.step(current_action)
