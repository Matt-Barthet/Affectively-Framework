from affectively.environments.heist import HeistEnvironment
import numpy as np


class HeistEnvironmentGameObs(HeistEnvironment):

    def __init__(self, id_number, graphics, weight, discretize=False, cluster=0, target_arousal=0, period_ra=False):
        self.gridWidth = 9
        self.gridHeight = 9
        self.elementSize = 0.5
        super().__init__(id_number=id_number, graphics=graphics, 
                         obs={"low": -np.inf, "high": np.inf, "shape": (947,), "type": np.float32},
                         weight=weight, frame_buffer=False, cluster=cluster, targetArousal=target_arousal, period_ra=period_ra)

    def construct_state(self, state):
        grid = np.asarray(state[0])
        state = np.asarray(state[1])
        one_hot = self.one_hot_encode(grid, 4)
        flattened_matrix_obs = [vector for sublist in one_hot for item in sublist for vector in item]
        combined_observations = list(flattened_matrix_obs) + list(state)
        return combined_observations