import numpy as np
from stable_baselines3 import PPO

from affectively.environments.base import compute_confidence_interval
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs

if __name__ == "__main__":

    run = 0
    preference_task = True
    classification_task = False
    weight = 0

    model_path = './ppo_solid_optimize_1.zip'

    env = SolidEnvironmentGameObs(0, graphics=True, weight=weight, logging=False, path="../Builds/MS_Solid/Racing.exe")
    model = PPO("MlpPolicy", tensorboard_log="../Tensorboard", device='cpu', env=env)
    model.load(model_path)
    model.set_parameters(model_path)

    arousal, scores = [], []
    for _ in range(30):
        state = env.reset()
        for i in range(600):
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()

        arousal.append(np.mean(env.arousal_trace))
        scores.append(env.best_score)
        env.best_score = 0
        env.arousal_trace.clear()

    print(f"Best Score: {compute_confidence_interval(scores)}, Mean Arousal: {compute_confidence_interval(arousal)}")
    env.close()

