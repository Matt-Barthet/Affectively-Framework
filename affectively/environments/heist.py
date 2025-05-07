from affectively.environments.base import BaseEnvironment
import numpy as np


class HeistEnvironment(BaseEnvironment):

    def __init__(self, id_number, graphics, weight, obs, logging=True, log_prefix="", targetArousal=0, frame_buffer=False, cluster=0, period_ra=False, args=None):
        args = ["-frameBuffer", f"{frame_buffer}"] if args is None else args +  ["-frameBuffer", f"{frame_buffer}"]
        self.frameBuffer = frame_buffer
        super().__init__(id_number=id_number, game='fps', graphics=graphics, obs_space=obs, args=args,
                         capture_fps=10, time_scale=1, weight=weight, logging=logging, log_prefix=log_prefix, cluster=cluster, target_arousal=targetArousal, period_ra=period_ra)

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
        state, reward, d, info = super().step(transformed_action)
        state = self.construct_state(state)
        self.reset_condition()
        return state, reward, d, info

    def handle_level_end(self):
        print("End of level reached, resetting environment.")
        self.reset()
        self.customSideChannel.levelEnd = False
