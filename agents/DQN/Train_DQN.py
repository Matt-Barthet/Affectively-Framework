import numpy as np
import sys
from examples.Agents.DQN.Rainbow_DQN import RainbowAgent, train
from examples.Agents.DQN.Rainbow_DQN_Resnet import RainbowResnetAgent


def main(run, weight, env_type):
    np.set_printoptions(suppress=True, precision=6)

    preference_task = True
    classification_task = False

    sideChannel = env.customSideChannel
    env.targetSignal = np.ones

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'

    agent = RainbowResnetAgent(env.observation_space.shape[0], env.action_space.nvec.tolist())
    num_episodes = 16638
    batch_size = 64
    update_target_every = 600
    train(agent, env, num_episodes, batch_size, update_target_every, name=f"run{run}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <run-number> <weight> <environment>")

    try:
        run = int(sys.argv[1])
        weight = float(sys.argv[2])
        env_type = sys.argv[3]
    except ValueError as e:
        print(f"Error in argument parsing: {e}")
        run = 1
        weight = 0
        env_type = "solid_cv"

    main(run, weight, env_type)

