import numpy as np
from affectively.environments.base import BaseEnvironment

class PiratesEnvironment(BaseEnvironment):

    def __init__(self, id_number, graphics, weight, obs, frame_buffer, cluster, period_ra, target_arousal, args=None, classifier=True, preference=True, capture_fps=60, decision_period=10):
        args = ["-frameBuffer", f"{frame_buffer}"] if args is None else args +  ["-frameBuffer", f"{frame_buffer}"]
        self.frameBuffer = frame_buffer
        time_scale = 5 if np.sign(capture_fps) > 0 else 1
        super().__init__(id_number=id_number, game='platform', graphics=graphics, obs_space=obs, args=args,
                         capture_fps=capture_fps, time_scale=time_scale, weight=weight, cluster=cluster,
                         period_ra=period_ra, target_arousal=target_arousal, classifier=classifier, preference=preference, decision_period=decision_period)

    def reset_condition(self):
        if self.customSideChannel.levelEnd:
            self.handle_level_end()
        if self.episode_length > 6000 / self.decision_period:
            self.reset()

    def reset(self, **kwargs):
        state = super().reset()
        state = self.construct_state(state)
        return state

    def step(self, action):

        save_load = 0
        
        # Saving and loading:
        # To save a state, assign an integer with a negative value.
        # To load that state, use the same integer with a positive value.
        # For example:
        # save_load = -12  # to save state 12
        # save_load = 12   # to load state 12

        transformed_action = (action[0] - 1, action[1], save_load,)
        state, reward, d, info = super().step(transformed_action)
        state = self.construct_state(state)
        self.reset_condition()
        return state, reward, d, info

    def handle_level_end(self):
        print("End of level reached, resetting environment.")
        self.reset()
        self.customSideChannel.levelEnd = False
