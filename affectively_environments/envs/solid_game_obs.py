import numpy as np

from .base import BaseEnvironment
from .solid import SolidEnvironment


class SolidEnvironmentGameObs(SolidEnvironment):

    def __init__(self, id_number, graphics, weight, path, logging=True, log_prefix=""):
        super().__init__(id_number=id_number, graphics=graphics,
                         obs={"low": -np.inf, "high": np.inf, "shape": (51,), "type": np.float32},
                         path=path, weight=weight, logging=logging, frame_buffer=False, log_prefix=log_prefix)

    def construct_state(self, state):
        if self.frameBuffer:
            game_obs = state[1]
        else:
            game_obs = state[0]
        self.game_obs = self.tuple_to_vector(game_obs)
        return self.game_obs
