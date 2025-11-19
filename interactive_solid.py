import os
import time
import numpy as np
import pygame
from affectively.environments.base import compute_confidence_interval
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.utils.logging import InteractiveDashboard

# Optional: run headless (no visible Pygame window)
# os.environ["SDL_VIDEODRIVER"] = "dummy"

# --- Global state shared with the gamepad and keyboard input ---
current_action = [1, 1]  # [horizontal, vertical]
running = True
accelerating = False
braking = False

# --- Deadzone Configuration ---
DEADZONE = 0.5  # Threshold for deadzone (adjustable)

# --- Initialize pygame and gamepad ---
pygame.init()
pygame.joystick.init()

# Initialize joystick at startup
joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Gamepad connected: {joystick.get_name()}")
    print(f"Number of buttons: {joystick.get_numbuttons()}")
    print(f"Number of axes: {joystick.get_numaxes()}")
else:
    print("No gamepad connected. Using keyboard only.")

# Check if there are any gamepads connected
if pygame.joystick.get_count() == 0:
    print("No gamepad connected.")
    running = False

# --- Main simulation ---
if __name__ == "__main__":

    env = SolidEnvironmentGameObs(
        0,
        graphics=True,
        weight=0,
        discretize=False,
        cluster=1,
        decision_period=1,
        target_arousal=1,
        period_ra=False,
        classifier=False,
        preference=False,
        capture_fps=-60,
    )
    env.callback = InteractiveDashboard(env)

    arousal, scores = [], []
    state = env.reset()

    env.episode_length = 6000
    pause = False
    end = False
    while True:

        # Check if there are any gamepads connected
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
            # print(end)
            if end or env.episode_length == 6000:

                env.callback.on_episode_end()
                env.callback.waiting_restart = True
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


        # Handle events (to capture ESC press and quit)
        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN:
                # Keyboard input for horizontal movement (arrows)
                if event.key == pygame.K_LEFT:  # Left arrow
                    current_action[0] = 0  # Left
                    print("← pressed (Keyboard)")
                elif event.key == pygame.K_RIGHT:  # Right arrow
                    current_action[0] = 2  # Right
                    print("→ pressed (Keyboard)")
                elif event.key == pygame.K_UP:  # Up arrow for acceleration
                    accelerating = True
                    print("Accelerate (Up arrow pressed)")
                elif event.key == pygame.K_DOWN:  # Down arrow for braking
                    braking = True
                    print("Brake (Down arrow pressed)")

            elif event.type == pygame.KEYUP:
                # Reset actions when arrow keys are released
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    current_action[0] = 1  # Neutral horizontal
                    print("Horizontal (Keyboard) released")
                if event.key == pygame.K_UP:
                    accelerating = False
                    print("Accelerate released")
                elif event.key == pygame.K_DOWN:
                    braking = False
                    print("Brake released")

            # Event-driven gamepad input
            if event.type == pygame.JOYBUTTONDOWN:
                print(event.button)
                if event.button == 0:  # A button (usually for accelerate)
                    accelerating = True
                    print("Accelerate (A pressed)")
                elif event.button == 2:  # X button (usually for brake)
                    braking = True
                    print("Brake (X pressed)")

                # D-pad input for horizontal direction
                if event.button == 13:  # D-pad Left (button 14)
                    current_action[0] = 0  # Move left (Gamepad D-pad Left)
                    print("← pressed (Gamepad D-pad Left)")

                elif event.button == 14:  # D-pad Right (button 15)
                    current_action[0] = 2  # Move right (Gamepad D-pad Right)
                    print("→ pressed (Gamepad D-pad Right)")


            elif event.type == pygame.JOYBUTTONUP:
                if event.button == 0:  # A button
                    accelerating = False
                    print("A released")
                elif event.button == 2:  # X button
                    braking = False
                    print("X released")

                # Reset D-pad inputs when released
                if event.button == 13 or event.button == 14:  # D-pad Left or Right
                    current_action[0] = 1  # Neutral horizontal
                    print("D-pad (Horizontal) released")

                if event.button == 6:
                    pause = not pause
                    print(f"Pause toggled to {pause}")
                    if pause:
                        env.callback.on_pause()
                    continue



        # Handle vertical movement for acceleration and braking
        if accelerating:
            current_action[1] = 2  # Accelerate
        elif braking:
            current_action[1] = 0  # Brake
        elif not accelerating and not braking:
            current_action[1] = 1  # Neutral vertical

        # Simulate environment step
        state, reward, terminated, info = env.step(current_action)
