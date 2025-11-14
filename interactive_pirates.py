import os
import time
import numpy as np
from pynput import keyboard
from affectively.environments.base import compute_confidence_interval
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.utils.logging import InteractiveDashboard

# Optional: run headless (no visible Pygame window)
os.environ["SDL_VIDEODRIVER"] = "dummy"

# --- Global state shared with the keyboard listener ---
current_action = [1, 1]  # [horizontal, vertical]
running = True


# --- Keyboard listener functions ---
def on_press(key):
    global current_action, running
    try:
        if key == keyboard.Key.left:
            current_action[0] = 0
            print("← pressed")
        elif key == keyboard.Key.right:
            current_action[0] = 2
            print("→ pressed")
        elif key == keyboard.Key.space:
            current_action[1] = 1
            print("↑ pressed")
        elif key == keyboard.Key.esc:
            print("ESC pressed — exiting")
            running = False
            return False  # stop listener
    except Exception as e:
        print(f"Key press error: {e}")


def on_release(key):
    global current_action
    try:
        if key in [keyboard.Key.left, keyboard.Key.right]:
            current_action[0] = 1  # neutral horizontal
        if key in [keyboard.Key.space]:
            current_action[1] = 0  # neutral vertical
    except Exception:
        pass


# --- Main simulation ---
if __name__ == "__main__":
    # Start global keyboard listener (runs in background thread)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    env = PiratesEnvironmentGameObs(
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
    env.callback = InteractiveDashboard(env, max_len=12)

    arousal, scores = [], []
    state = env.reset()

    print("Simulation started. Use arrow keys for input, ESC to quit.")
    print("You can switch windows — keyboard input still works in background.")

    while running:
        state, reward, terminated, info = env.step(current_action)

        if terminated:
            arousal.append(np.mean(env.arousal_trace))
            scores.append(env.best_score)
            env.best_score = 0
            env.arousal_trace.clear()
            state = env.reset()
            print(
                f"Episode done → Mean Arousal: {arousal[-1]:.3f}, Score: {scores[-1]:.3f}"
            )


    # --- Cleanup ---
    listener.stop()
    env.close()
    print(f"Best Score: {compute_confidence_interval(scores)}, "
          f"Mean Arousal: {compute_confidence_interval(arousal)}")
