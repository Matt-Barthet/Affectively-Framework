import numpy as np
from affectively.environments.base import BaseEnvironment

class PiratesEnvironment(BaseEnvironment):

    def __init__(self, id_number, graphics, weight, obs, frame_buffer, cluster, period_ra, target_arousal, args=None):
        args = ["-frameBuffer", f"{frame_buffer}"] if args is None else args +  ["-frameBuffer", f"{frame_buffer}"]
        self.frameBuffer = frame_buffer
        super().__init__(id_number=id_number, game='platform', graphics=graphics, obs_space=obs, args=args,
                         capture_fps=60, time_scale=5, weight=weight, cluster=cluster,
                         period_ra=period_ra, target_arousal=target_arousal)

    def reset_condition(self):
        if self.customSideChannel.levelEnd:
            self.handle_level_end()
        if self.episode_length > 600:
            self.episode_length = 0
            self.reset()

    def reset(self, **kwargs):
        state = super().reset()
        state = self.construct_state(state)
        return state

    def step(self, action):
        transformed_action = (action[0] - 1, action[1], 0,)
        state, reward, d, info = super().step(transformed_action)
        state = self.construct_state(state)
        self.reset_condition()
        return state, reward, d, info

    def handle_level_end(self):
        print("End of level reached, resetting environment.")
        self.reset()
        self.customSideChannel.levelEnd = False
