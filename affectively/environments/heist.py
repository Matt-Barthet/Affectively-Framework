from random import random
from affectively.environments.base import BaseEnvironment


class HeistEnvironment(BaseEnvironment):

    def __init__(self, id_number, graphics, weight, obs, targetArousal, frame_buffer, cluster, period_ra, args=None, classifier=True, preference=True, decision_period=10, capture_fps=10, sensitivity=1, imitate=False, correct_step_bug=True):
        args = ["-frameBuffer", f"{frame_buffer}", "-sensitivity", f"{sensitivity}"] if args is None else args +  ["-frameBuffer", f"{frame_buffer}", "-sensitivity", f"{sensitivity}"]
        self.frameBuffer = frame_buffer
        super().__init__(id_number=id_number, game='fps', graphics=graphics, obs_space=obs, args=args,
                         capture_fps=capture_fps, time_scale=1, weight=weight, cluster=cluster, target_arousal=targetArousal, period_ra=period_ra, classifier=classifier, preference=preference,
                         decision_period=decision_period, imitate=imitate, correct_step_bug=correct_step_bug)

    def reset(self, **kwargs):
        state = super().reset()
        return self.construct_state(state)

    def sample_weighted_action(self):
        movementlr_weights = [27.4, 46.2, 26.4]
        movement_fb_weights = [5.5, 37.8, 56.7]
        shooting_weights = [26.3, 73.7]

        movementlr_options = [0, 1, 2]
        movement_fb_options = [0, 1, 2]
        shooting_options = [0, 1]

        action = self.action_space.sample()
        action[3] = random.choices(movementlr_options, weights=movementlr_weights)[0]
        action[2] = random.choices(movement_fb_options, weights=movement_fb_weights)[0]
        action[4] = random.choices(shooting_options, weights=shooting_weights)[0]
        return action


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
