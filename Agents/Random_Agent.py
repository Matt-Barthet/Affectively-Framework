from Environments.Solid_Environment import Solid_Environment
from Environments.Heist_Environment import Heist_Environment
from Environments.Pirates_Environment import Pirates_Environment


if __name__ == "__main__":

    weight = 0.5
    env = Pirates_Environment(0, graphics=True, weight=weight, logging=False)

    for _ in range(30):
        _ = env.reset()
        for i in range(600):
            action = env.action_space.sample()
            _, reward, done, info = env.step(action)
            if done:
                state = env.reset()
    env.close()
