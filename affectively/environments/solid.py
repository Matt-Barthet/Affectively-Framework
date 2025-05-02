import random
import numpy as np
from affectively_environments.envs.base import BaseEnvironment


class SolidEnvironment(BaseEnvironment):

    def __init__(self, id_number, graphics, weight, path, obs, logging=True, frame_buffer=False, args=None, log_prefix=""):
        if args is None:
            args = []
        self.frameBuffer = frame_buffer
        args += ["-frameBuffer", f"{frame_buffer}"]
        super().__init__(id_number=id_number, game='Solid', graphics=graphics, obs_space=obs, path=path, args=args,
                         capture_fps=5, time_scale=1, weight=weight, logging=logging, log_prefix=log_prefix)

    def sample_action(self):
        return self.action_space.sample()

    def sample_weighted_action(self):
        steering_distribution = [0, 0, 0, 0, 1, 1, 1, -1, -1, -1]
        pedal_distribution = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1]
        action = self.action_space.sample()
        action[0] = random.choice(steering_distribution)
        action[1] = random.choice(pedal_distribution)
        return action

    def calculate_reward(self):
        self.current_reward = np.clip((self.current_score - self.previous_score), 0, 1)
        self.cumulative_reward += self.current_reward
        self.best_cumulative_reward = self.current_reward if self.current_reward > self.best_cumulative_reward else self.best_cumulative_reward

    def reset_condition(self):
        if self.episode_length > 600:
            self.episode_length = 0
            self.reset()

    def reset(self, **kwargs):
        state = super().reset()
        state = self.construct_state(state)
        return state

    def step(self, action):
        transformed_action = np.asarray([tuple([action[0] - 1, action[1] - 1])])
        state, env_score, d, info = super().step(transformed_action)
        arousal = 0
        state = self.construct_state(state)
        self.calculate_reward()
        self.reset_condition()
        final_reward = self.current_reward * (1 - self.weight) + (arousal * self.weight)
        return state, final_reward, d, info
