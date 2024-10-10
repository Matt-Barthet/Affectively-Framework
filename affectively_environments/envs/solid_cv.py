import numpy as np
from matplotlib import pyplot as plt

from .base import BaseEnvironment
from gym import spaces

from .solid import SolidEnvironment


class SolidEnvironmentCV(SolidEnvironment):

    def __init__(self, id_number, graphics, weight, path, logging=True):
        super().__init__(id_number=id_number, graphics=graphics,
                         obs={"low": 0, "high": 255, "shape": (75, 100, 1), "type": np.uint8},
                         path=path, weight=weight, frame_buffer=True, logging=logging)

    def construct_state(self, state):
        visual_buffer = np.asarray(state[0]) * 255
        self.game_obs = self.tuple_to_vector(state[1])
        return np.asarray(visual_buffer, dtype=np.uint8)
