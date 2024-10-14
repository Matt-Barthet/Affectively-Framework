import numpy as np

from affectively_environments.envs.solid_game_obs import SolidEnvironmentGameObs
from examples.Agents.Rainbow_DQN import RainbowAgent, train

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=6)

    run = 1
    preference_task = True
    classification_task = False
    weight = 0

    env = SolidEnvironmentGameObs(id_number=run, weight=weight, graphics=True, logging=True, path="../Builds/MS_Solid/Racing.exe", log_prefix="DQN-")
    sideChannel = env.customSideChannel
    env.targetSignal = np.ones

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'

    agent = RainbowAgent(51, env.action_space.nvec.tolist())
    num_episodes = 16700
    batch_size = 64
    update_target_every = 600
    train(agent, env, num_episodes, batch_size, update_target_every)

