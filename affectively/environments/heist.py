from affectively.environments.base import BaseEnvironment
import numpy as np


class HeistEnvironment(BaseEnvironment):

    def __init__(self, id_number, graphics, weight, obs, logging=True, log_prefix="", targetArousal=0, frame_buffer=False, cluster=0, period_ra=False, args=None):

        self.death_applied = False
        self.previous_health = 0

        self.current_ammo = 0
        self.current_health = 0
        self.previous_angle, self.current_angle = 0, 0

        if args is None:
            args = []

        self.frameBuffer = frame_buffer
        args += ["-frameBuffer", f"{frame_buffer}"]
        super().__init__(id_number=id_number, game='fps', graphics=graphics, obs_space=obs, args=args,
                         capture_fps=15, time_scale=3, weight=weight, logging=logging, log_prefix=log_prefix)

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

    def step(self, action):
        transformed_action = [
            action[0] * 4,
            action[1] * 2,
            np.round(action[2]+1),
            np.round(action[3]+1),
            np.round(action[4]/2 + 0.5),
            0
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
