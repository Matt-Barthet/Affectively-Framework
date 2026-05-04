from affectively.environments.heist import HeistEnvironment
import numpy as np
import random

class HeistEnvironmentGameObs(HeistEnvironment):

    def __init__(self, id_number, graphics, weight, discretize=False, cluster=0, target_arousal=0, period_ra=False, classifier=True, preference=True, decision_period=10, capture_fps=10,sensitivity=1, imitate=False, correct_step_bug=True):
        self.gridWidth = 5
        self.gridHeight = 5
        self.elementSize = 1

        super().__init__(id_number=id_number, graphics=graphics, 
                         obs={"low": -np.inf, "high": np.inf, "shape": (152,), "type": np.float32},
                         weight=weight, frame_buffer=False, cluster=cluster, targetArousal=target_arousal, period_ra=period_ra, classifier=classifier, preference=preference,
                         capture_fps=capture_fps, decision_period=decision_period, sensitivity=sensitivity,correct_step_bug=correct_step_bug)
        self.discretize = discretize

    def sample_weighted_action(self):
        movement_options = [0, 1, 2]
        movement_weights = [27.4, 46.2, 26.4]

        forward_options = [0, 1, 2]
        forward_weights = [5.5, 37.8, 56.7]

        shooting_options = [0, 1]
        shooting_weights = [26.3, 73.7]

        action = self.action_space.sample()
        action[2] = random.choices(forward_options, weights=forward_weights)[0]
        action[3] = random.choices(movement_options, weights=movement_weights)[0]
        action[4] = random.choices(shooting_options, weights=shooting_weights)[0]
        return action

    def construct_state(self, state):
        grid = np.asarray(state[0])
        state = np.asarray(state[1])
        if self.discretize:
            state = self.discretize_observations(state)
            return state
        one_hot = self.one_hot_encode(grid, 4)
        flattened_matrix_obs = [vector for sublist in one_hot for item in sublist for vector in item]
        combined_observations = list(flattened_matrix_obs) + list(state)
        return combined_observations

    def discretize_observations(self, game_obs):

        position = game_obs[0:3]
        position_discrete = np.round(position / 30)
        # print(position_discrete)
        position_discrete[0] = 0 if position_discrete[0] == -0 else position_discrete[0]
        position_discrete[1] = 0 if position_discrete[1] == -0 else position_discrete[1]
        position_discrete[2] = 0 if position_discrete[2] == -0 else position_discrete[2]

        velocity = game_obs[8:10]
        velocity_discrete = np.round(np.linalg.norm(velocity) / 10)

        score_bin = np.round(self.current_score / 40)

        health = np.round(game_obs[3] / 30)
        mouse_x = np.round(game_obs[11] / 45)
        mouse_y = np.round(game_obs[12] / 45)
        rot_x = np.round(game_obs[13] / 45)
        rot_y = np.round(game_obs[14] / 45)

        discrete_obs = (
            list(position_discrete) +
            [
                velocity_discrete,
                score_bin,
                health,
                mouse_x, mouse_y, rot_x, rot_y
            ]
        )

        discrete_obs = np.array(discrete_obs)
        discrete_obs = np.where(discrete_obs == -0, 0, discrete_obs)

        return discrete_obs
