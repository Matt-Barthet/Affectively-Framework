from affectively.environments.heist import HeistEnvironment
import numpy as np


class HeistEnvironmentGameObs(HeistEnvironment):

    def __init__(self, id_number, graphics, weight, logging=True, log_prefix="", discretize=False):
        """ ---- Heist! specific code ---- """
        self.gridWidth = 9
        self.gridHeight = 9
        self.elementSize = 0.5
        super().__init__(id_number=id_number, graphics=graphics, 
                         obs={"low": -np.inf, "high": np.inf, "shape": (951,), "type": np.float32},
                         weight=weight, logging=logging, log_prefix=log_prefix, frame_buffer=False)


    def construct_state(self, state):
        grid = np.asarray(state[0])
        state = np.asarray(state[1])
        one_hot = self.one_hot_encode(grid, 4)
        flattened_matrix_obs = [vector for sublist in one_hot for item in sublist for vector in item]
        combined_observations = list(flattened_matrix_obs) + list(state)
        return combined_observations