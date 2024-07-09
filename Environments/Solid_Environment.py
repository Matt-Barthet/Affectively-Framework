from abc import ABC
import numpy as np
from Environments.BaseEnvironment import BaseEnvironment


class Solid_Environment(BaseEnvironment, ABC):

    def __init__(self, id_number, graphics, weight, logging=True):
        obs_space = {"low": -np.inf, "high": np.inf, "shape": (49, )}

        try:
            super().__init__(id_number=id_number, game='Solid', graphics=graphics, obs_space=obs_space,
                             path="./Builds/MS_SolidRally/Racing.exe", args=[], capture_fps=5, time_scale=1, weight=weight,
                             logging=logging)
        except:
            print("Windows build and default path (./Builds) failed, trying to load mac build.")
            super().__init__(id_number=id_number, game='Solid', graphics=graphics, obs_space=obs_space,
                         path="./Builds/SolidRally.app", args=[], capture_fps=5, time_scale=1, weight=weight,
                         logging=logging)

    def calculate_reward(self, state):
        rotation_component = 1 if (180 - np.abs(state[-1])) > 60 else -1
        speed_component = np.linalg.norm([state[0], state[1], state[2]]) / 80
        self.current_reward = (self.current_score - self.previous_score) + rotation_component * speed_component

    def reset_condition(self):
        if self.episode_length > 600:
            self.episode_length = 0
            self.reset()

    def reset(self, **kwargs):
        state = super().reset()
        return self.tuple_to_vector(state[0])

    def step(self, action):
        transformed_action = np.asarray([tuple([action[0] - 1, action[1] - 1])])
        state, env_score, arousal, d, info = super().step(transformed_action)
        state = self.tuple_to_vector(state[0])
        self.calculate_reward(state)
        self.cumulative_reward += self.current_reward
        self.best_cumulative_reward = self.current_reward if self.current_reward > self.best_cumulative_reward else self.best_cumulative_reward
        self.reset_condition()
        final_reward = self.current_reward * (1 - self.weight) + (arousal * self.weight)
        return state, final_reward, d, info
