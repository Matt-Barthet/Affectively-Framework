import random

import numpy as np
from affectively.environments.base import BaseEnvironment

class PiratesEnvironment(BaseEnvironment):

    def __init__(self, id_number, graphics, weight, obs, frame_buffer, cluster, period_ra, target_arousal, absolute=False, args=None, classifier=True, preference=True, capture_fps=60, decision_period=10):
        args = ["-frameBuffer", f"{frame_buffer}"] if args is None else args +  ["-frameBuffer", f"{frame_buffer}"]
        self.frameBuffer = frame_buffer
        time_scale = 5 if np.sign(capture_fps) > 0 else 1
        super().__init__(id_number=id_number, game='platform', graphics=graphics, obs_space=obs, args=args,
                         capture_fps=capture_fps, time_scale=time_scale, weight=weight, cluster=cluster, absolute=absolute,
                         period_ra=period_ra, target_arousal=target_arousal, classifier=classifier, preference=preference, decision_period=decision_period)

    def sample_weighted_action(self):
        movement_options = [0, 1, 2]
        movement_weights = [10.9, 47.2, 41.9]
        jump_options = [0, 1]
        jump_weights = [55.3, 44.7]
        action = self.action_space.sample()
        action[0] = random.choices(movement_options, weights=movement_weights)[0]
        action[1] = random.choices(jump_options, weights=jump_weights)[0]
        return action

    def reset(self, **kwargs):
        state = super().reset()
        state = self.construct_state(state)
        return state

    def step(self, action):
        transformed_action = list([action[0] - 1]) + list(action[1:])
        state, reward, d, info = super().step(transformed_action)
        state = self.construct_state(state)
        return state, reward, d, info

    def handle_level_end(self):
        print("End of level reached, resetting environment.")
        self.reset()
        self.customSideChannel.levelEnd = False
