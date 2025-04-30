from abc import ABC
import configparser

import mlagents_envs.exception
import numpy as np

from torch.utils.tensorboard import SummaryWriter

import numpy as np

from affectively_environments.envs.solid_game_obs import SolidEnvironmentGameObs
from examples.Agents.Explore.Explore import Explorer

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=6)

    run = 5
    preference_task = True
    classification_task = False
    weight = 0

    env = SolidEnvironmentGameObs(id_number=run, weight=weight, graphics=True, logging=True,
                                  path="../Builds/MS_Solid/Racing.exe", log_prefix="Go-Explore/", discretize=True,)
    sideChannel = env.customSideChannel
    env.targetSignal = np.ones

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'

    agent = Explorer(env, f"Solid-optimize-{run}-archive")
    agent.explore(f"run{run}", 500000, 50)
