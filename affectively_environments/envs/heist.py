from affectively_environments.envs.base import BaseEnvironment
import numpy as np


class HeistEnvironment(BaseEnvironment):

    def __init__(self, id_number, graphics, weight, logging=True, log_prefix=""):

        """ ---- Heist! specific code ---- """
        self.gridWidth = 9
        self.gridHeight = 9
        self.elementSize = 0.5
        self.death_applied = False
        self.previous_health = 0

        self.current_ammo = 0
        self.current_health = 0
        self.previous_angle, self.current_angle = 0, 0

        super().__init__(id_number=id_number, graphics=graphics,
                         obs_space={"low": -np.inf, "high": np.inf, "shape": (341,)},
                         args=["-agentGridWidth", f"{self.gridWidth}", "-agentGridHeight", f"{self.gridHeight}",
                               "-cellSize", f"{self.elementSize}"], capture_fps=15, time_scale=1, weight=weight,
                         game='Heist', logging=logging, log_prefix=log_prefix)

    def calculate_reward(self):
        self.current_reward = np.clip((self.current_score - self.previous_score), 0, 1)
        self.cumulative_reward += self.current_reward
        self.best_cumulative_reward = self.current_reward if self.current_reward > self.best_cumulative_reward else self.best_cumulative_reward

    def reset_condition(self):
        if self.episode_length > 600:
            self.reset()
        if self.customSideChannel.levelEnd:
            self.handle_level_end()

    def reset(self, **kwargs):
        state = super().reset()
        return self.construct_state(state)

    def construct_state(self, state):
        grid = np.asarray(state[0])
        state = np.asarray(state[1])
        one_hot = self.one_hot_encode(grid, 4)
        flattened_matrix_obs = [vector for sublist in one_hot for item in sublist for vector in item]
        combined_observations = list(flattened_matrix_obs) + list(state[3:])
        return combined_observations

    def step(self, action):
        transformed_action = [
            action[0] * 4,
            action[1] * 2,
            np.round(action[2]+1),
            np.round(action[3]+1),
            np.round(action[4]/2 + 0.5),
        ]
        state, env_score, arousal, d, info = super().step(transformed_action)
        state = self.construct_state(state)
        self.calculate_reward()
        self.reset_condition()
        final_reward = self.current_reward * (1 - self.weight) + (arousal * self.weight)
        return state, final_reward, d, info


    def handle_level_end(self):
        print("End of level reached, resetting environment.")
        self.reset()
        self.customSideChannel.levelEnd = False
