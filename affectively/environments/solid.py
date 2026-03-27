import random
import numpy as np
from affectively.environments.base import BaseEnvironment


class SolidEnvironment(BaseEnvironment):

    def __init__(self, id_number, graphics, weight, obs, frame_buffer, cluster, target_arousal, period_ra, args=None, classifier=True, preference=True, decision_period=10, capture_fps=5, imitate=False):
        args = ["-frameBuffer", f"{frame_buffer}"] if args is None else args +  ["-frameBuffer", f"{frame_buffer}"]
        self.frameBuffer = frame_buffer
        super().__init__(id_number=id_number, game='Solid', graphics=graphics, obs_space=obs, args=args,
                         capture_fps=capture_fps, time_scale=1, weight=weight, cluster=cluster,
                         target_arousal=target_arousal, period_ra=period_ra, classifier=classifier, preference=preference, decision_period=decision_period, imitate=imitate)

    def sample_weighted_action(self):
        steering_options = [0, 1, 2]
        steering_weights = [26.0, 56.8, 17.2]
        pedal_options = [0, 1, 2]
        pedal_weights = [0.3, 37.3, 62.4]
        action = self.action_space.sample()
        action[0] = random.choices(steering_options, weights=steering_weights)[0]
        action[1] = random.choices(pedal_options, weights=pedal_weights)[0]
        return action

    def reset(self, **kwargs):
        state = super().reset()
        state = self.construct_state(state)
        return state

    def step(self, action):
        transformed_action = np.asarray((action[0], action[1], action[2]))
        state, reward, d, info = super().step(transformed_action)
        state = self.construct_state(state)
        return state, reward, d, info
