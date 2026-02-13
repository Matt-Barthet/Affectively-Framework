from affectively.environments.base import BaseEnvironment
import numpy as np


class HeistEnvironment(BaseEnvironment):

    def __init__(self, id_number, graphics, weight, obs, targetArousal, frame_buffer, cluster, period_ra, args=None, classifier=True, preference=True, decision_period=10, capture_fps=10, sensitivity=1):
        args = ["-frameBuffer", f"{frame_buffer}", "-sensitivity", f"{sensitivity}"] if args is None else args +  ["-frameBuffer", f"{frame_buffer}", "-sensitivity", f"{sensitivity}"]
        self.frameBuffer = frame_buffer
        super().__init__(id_number=id_number, game='fps', graphics=graphics, obs_space=obs, args=args,
                         capture_fps=capture_fps, time_scale=1, weight=weight, cluster=cluster, target_arousal=targetArousal, period_ra=period_ra, classifier=classifier, preference=preference,
                         decision_period=decision_period)

    def reset(self, **kwargs):
        state = super().reset()
        return self.construct_state(state)

    def step(self, action):
        transformed_action = [
            action[0],
            action[1],
            action[2] - 1,
            action[3]- 1,
            action[4],
            action[5]
        ]
        state, reward, done, info = super().step(transformed_action)
        state = self.construct_state(state)
        return state, reward, done, info