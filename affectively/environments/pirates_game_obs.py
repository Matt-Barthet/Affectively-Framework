import numpy as np
from affectively.environments.pirates import PiratesEnvironment

class PiratesEnvironmentGameObs(PiratesEnvironment):

    def __init__(self, id_number, graphics, weight, discretize, cluster, period_ra, target_arousal, classifier=True, preference=True):

        """ ---- Pirates! specific code ---- """
        self.gridWidth = 11
        self.gridHeight = 11
        self.elementSize = 1
        super().__init__(id_number=id_number, graphics=graphics, 
                         obs={"low": -np.inf, "high": np.inf, "shape": (381,), "type": np.float32},
                         weight=weight, frame_buffer=False, cluster=cluster,
                         period_ra=period_ra, target_arousal=target_arousal, classifier=classifier, preference=preference)

    def construct_state(self, state):
        grid = state[0]
        state = state[1]
        one_hot = self.one_hot_encode(grid, 7)
        flattened_matrix_obs = [vector for sublist in one_hot for item in sublist for vector in item]
        combined_observations = list(state) + list(flattened_matrix_obs)
        return np.asarray(combined_observations)