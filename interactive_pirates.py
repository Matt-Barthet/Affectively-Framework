import os
import numpy as np
import pygame
from affectively.environments.base import compute_confidence_interval
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.utils.logging import InteractiveDashboard

# Headless mode (keep this if needed)
# os.environ["SDL_VIDEODRIVER"] = "dummy"

# --- Global state ---
current_action = [1, 0]  # [horizontal, jump]
jumping = False

# --- Deadzone Configuration ---
DEADZONE = 0.5  # Threshold for deadzone (adjustable)

# --- Initialize pygame and gamepad ---
pygame.init()

# Check if there are any gamepads connected
if pygame.joystick.get_count() == 0:
    print("No gamepad found.")
    joystick = None
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Gamepad initialized:", joystick.get_name())

KEYBOARD_ENABLED = ("dummy" not in os.environ.get("SDL_VIDEODRIVER", ""))

# --- Main simulation ---
if __name__ == "__main__":

    env = PiratesEnvironmentGameObs(
        0,
        graphics=True,
        weight=0,
        discretize=False,
        cluster=0,
        decision_period=1,
        target_arousal=1,
        period_ra=False,
        classifier=False,
        preference=False,
        capture_fps=-60,
    )

    env.callback = InteractiveDashboard(env)
    state = env.reset()

    pause = True
    end = False
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
                    print(event.button)
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


            if KEYBOARD_ENABLED:
                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_LEFT:
                        current_action[0] = 0
                        print("← (keyboard)")
                    elif event.key == pygame.K_RIGHT:
                        current_action[0] = 2
                        print("→ (keyboard)")
                    elif event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                        jumping = True
                        print("Jump (keyboard)")

                elif event.type == pygame.KEYUP:
                    if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        current_action[0] = 1
                        print("Horizontal released (keyboard)")
                    if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                        jumping = False
                        print("Jump released (keyboard)")


            if event.type == pygame.JOYBUTTONDOWN:

                # A → jump
                if event.button == 0:
                    jumping = True
                    print("A pressed (jump)")

                # D-pad Left / Right
                if event.button == 13:
                    current_action[0] = 0
                    print("← (D-pad)")
                elif event.button == 14:
                    current_action[0] = 2
                    print("→ (D-pad)")

            elif event.type == pygame.JOYBUTTONUP:
                if event.button == 0:
                    jumping = False
                    print("A released")

                if event.button in (13, 14):
                    current_action[0] = 1
                    print("D-pad released")

                if event.button == 6:
                    pause = not pause
                    print(f"Pause toggled to {pause}")
                    if pause:
                        env.callback.on_pause()
                    continue


        if joystick is not None:

            # Get the axes (analog stick for horizontal movement)
            x_axis = joystick.get_axis(0)  # Left/right horizontal (X axis)

            # Apply deadzone logic
            if abs(x_axis) < DEADZONE:
                x_axis = 0  # Ignore small movements within the deadzone

            # Update horizontal movement based on joystick
            if abs(x_axis) > DEADZONE:
                # Joystick moves the character
                if x_axis < 0:  # Joystick left
                    current_action[0] = 0
                    print("Joystick left")
                elif x_axis > 0:  # Joystick right
                    current_action[0] = 2
                    print("Joystick right")
            else:
                if current_action[0] != 1:
                    pass

            if joystick.get_button(0):
                jumping = True

            try:
                if joystick.get_button(13):  
                    current_action[0] = 0
                elif joystick.get_button(14): 
                    current_action[0] = 2
                elif abs(x_axis) <= DEADZONE:
                    current_action[0] = 1
            except:
                if abs(x_axis) <= DEADZONE:
                    current_action[0] = 1

        current_action[1] = 1 if jumping else 0
        state, reward, terminated, info = env.step(current_action)