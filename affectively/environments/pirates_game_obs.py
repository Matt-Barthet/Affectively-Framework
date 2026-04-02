import numpy as np

from affectively.environments.pirates import PiratesEnvironment

class PiratesEnvironmentGameObs(PiratesEnvironment):

    def __init__(self, id_number, graphics, weight, discretize, cluster, period_ra, target_arousal, classifier=True, preference=True, capture_fps=60, decision_period=10, imitate=False):
        self.gridWidth = 3 if discretize else 7
        self.gridHeight = 3 if discretize else 7
        self.elementSize = 3
        print(self.gridWidth, self.gridHeight, self.elementSize)
        if not (
                discretize):
            obs_shape = (9 + self.gridWidth * self.gridHeight * 5,)
        else:
            obs_shape = (2 + 1 + self.gridWidth * self.gridHeight * 2,)

        super().__init__(id_number=id_number, graphics=graphics,
                         obs={"low": -np.inf, "high": np.inf, "shape": obs_shape, "type": np.float32},
                         weight=weight, frame_buffer=False, cluster=cluster, absolute=discretize==1,
                         period_ra=period_ra, target_arousal=target_arousal, classifier=classifier, preference=preference,
                         args=['-gridWidth', f"{self.gridWidth}", '-gridHeight', f"{self.gridHeight}", '-reloadEvery', f"{3 if discretize else 10}", '-relativeObs', 'True' if discretize else 'False'], capture_fps=capture_fps, decision_period=decision_period)
        """ ---- Pirates! specific code ---- """
        self.discretize = discretize
        self.estimated_position = [0, 0]

    def reset(self, **kwargs):
        self.estimated_position = [0, 0]
        return super().reset(**kwargs)

    def construct_state(self, state):
        grid, other_state = state
        if not self.discretize:
            one_hot = self.one_hot_encode(grid, 5)
            flattened_matrix_obs = one_hot.reshape(-1)
            combined_observations = np.concatenate((np.asarray(other_state).ravel(), flattened_matrix_obs))
        else:
            combined_observations = self.discretize_observations(other_state, grid)
        return combined_observations

    def discretize_observations(self, game_obs, matrix):
        self.raw_state = list(game_obs)
        position_discrete = np.round( np.asarray([game_obs[0], game_obs[2]]) / 7.5)
        position_discrete[(position_discrete == -0)] = 0
        # flattened_matrix = matrix.flatten()
        # flattened_matrix[(flattened_matrix != 5)] = 0
        discrete_obs = list(position_discrete) # + list(flattened_matrix)
        return discrete_obs
