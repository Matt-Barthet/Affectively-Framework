import numpy as np

from .base import BaseEnvironment


class PiratesEnvironment(BaseEnvironment):

    def __init__(self, id_number, graphics, weight, path, logging=True, log_prefix=""):

        """ ---- Pirates! specific code ---- """
        self.gridWidth = 11
        self.gridHeight = 11
        self.elementSize = 1

        self.last_x = -np.inf
        self.max_x = -np.inf
        self.death_applied = False
        self.previous_score = 0  # maximum possible score of 460

        super().__init__(id_number=id_number, graphics=graphics,
                         obs_space={"low": -np.inf, "high": np.inf, "shape": (852,)},
                         path=path,
                         args=["-gridWidth", f"{self.gridWidth}", "-gridHeight", f"{self.gridHeight}",
                               "-elementSize",
                               f"{self.elementSize}"], capture_fps=60, time_scale=5, weight=weight, game='Pirates',
                         logging=logging, log_prefix=log_prefix)

    def calculate_reward(self):
        self.current_reward = np.clip((self.current_score - self.previous_score), 0, 1)
        self.cumulative_reward += self.current_reward
        self.best_cumulative_reward = self.current_reward if self.current_reward > self.best_cumulative_reward else self.best_cumulative_reward

    def reset_condition(self):
        if self.customSideChannel.levelEnd:
            self.handle_level_end()
        elif self.episode_length > 600:
            self.reset()

    def reset(self, **kwargs):
        state = super().reset()
        state = self.construct_state(state)
        return state

    def construct_state(self, state):
        grid = state[0]
        state = state[1]
        one_hot = self.one_hot_encode(grid, 7)
        flattened_matrix_obs = [vector for sublist in one_hot for item in sublist for vector in item]
        combined_observations = list(state[2:]) + list(flattened_matrix_obs)
        return combined_observations

    def step(self, action):
        transformed_action = (action[0] - 1, action[1])
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
