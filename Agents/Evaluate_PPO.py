import numpy as np
from stable_baselines3 import PPO

from Environments.BaseEnvironment import compute_confidence_interval
from Environments.Solid_Environment import Solid_Environment
from Environments.Heist_Environment import Heist_Environment
from Environments.Pirates_Environment import Pirates_Environment

if __name__ == "__main__":

    weight = 0
    model_path = ''

    env = Pirates_Environment(0, graphics=True, weight=weight, logging=False)
    model = PPO("MlpPolicy", env=env, tensorboard_log="../Tensorboard", device='cpu')
    model.load(model_path)
    model.set_parameters(model_path)

    arousal, scores = [], []
    for _ in range(30):
        state = env.reset()
        for i in range(600):
            action, _ = model.predict(state, deterministic=False)
            _, reward, done, info = env.step(action)
            if done:
                state = env.reset()

        arousal.append(np.mean(env.arousal_trace))
        scores.append(env.best_score)
        env.best_score = 0
        env.arousal_trace.clear()

    print(f"Best Score: {compute_confidence_interval(scores)}, Mean Arousal: {compute_confidence_interval(arousal)}")
    env.close()

