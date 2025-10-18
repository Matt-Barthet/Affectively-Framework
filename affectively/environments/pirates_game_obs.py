import numpy as np
from affectively.environments.pirates import PiratesEnvironment

class PiratesEnvironmentGameObs(PiratesEnvironment):

    def __init__(self, id_number, graphics, weight, discretize, cluster, period_ra, target_arousal, classifier=True, preference=True):

        """ ---- Pirates! specific code ---- """
        self.gridWidth = 7
        self.gridHeight = 7
        self.elementSize = 1
        super().__init__(id_number=id_number, graphics=graphics, 
                         obs={"low": -np.inf, "high": np.inf, "shape": (288,), "type": np.float32},
                         weight=weight, frame_buffer=False, cluster=cluster,
                         period_ra=period_ra, target_arousal=target_arousal, classifier=classifier, preference=preference,
                         args=['-gridWidth', f"{self.gridWidth}", '-gridHeight', f"{self.gridHeight}"])

    def construct_state(self, state):
        grid, other_state = state

        one_hot = self.one_hot_encode(grid, 5)  # suppose shape (H, W, 7)
        flattened_matrix_obs = one_hot.reshape(-1)  # flatten to 1D efficiently

        combined_observations = np.concatenate((np.asarray(other_state).ravel(), flattened_matrix_obs))
        return combined_observations