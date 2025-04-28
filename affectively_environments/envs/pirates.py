import numpy as np
from affectively_environments.envs.base import BaseEnvironment


class PiratesEnvironment(BaseEnvironment):

    def __init__(self, id_number, graphics, weight, logging=True, log_prefix="", targetArousal=0, cluster=0, period_ra=False):

        """ ---- Pirates! specific code ---- """
        self.gridWidth = 11
        self.gridHeight = 11
        self.elementSize = 1

        self.last_x = -np.inf
        self.max_x = -np.inf
        self.death_applied = False
        self.previous_score = 0
        self.surrogate_length = 33
        super().__init__(id_number=id_number, graphics=graphics,
                         obs_space={"low": -np.inf, "high": np.inf, "shape": (887,)},
                         args=["-gridWidth", f"{self.gridWidth}", "-gridHeight", f"{self.gridHeight}",
                               "-elementSize", f"{self.elementSize}"], capture_fps=60, time_scale=5,
                         weight=weight, game='Pirates', logging=logging, log_prefix=log_prefix,
                         target_arousal=targetArousal, cluster=cluster, period_ra=period_ra)


    def reset_condition(self):
        if self.customSideChannel.levelEnd:
            self.handle_level_end()
        elif self.episode_length > 600:
            self.reset()

    def reset(self, **kwargs):
        state = super().reset()
        state = self.construct_state(state)
        return state

    def construct_state(self, state):
        grid = state[0]
        state = state[1]
        one_hot = self.one_hot_encode(grid, 7)
        flattened_matrix_obs = [vector for sublist in one_hot for item in sublist for vector in item]
        combined_observations = list(state) + list(flattened_matrix_obs)
        # print(combined_observations)
        return np.asarray(combined_observations)

    def step(self, action):
        transformed_action = (action[0] - 1, action[1])
        state, env_score, d, info = super().step(transformed_action)

        self.surrogate_list.append(state[1][-self.surrogate_length:])
        state = self.construct_state(state)
        arousal, final_reward = 0, 0

        if self.arousal_episode_length % 15 == 0:
            self.generate_arousal()
            self.arousal_episode_length = 0
            self.surrogate_list.clear()

        if self.period_ra and (len(self.episode_arousal_trace) > 0):
            print("assigning reward asynchronously!")
            final_reward = self.reward_behavior() * (1 - self.weight) + (self.reward_affect() * self.weight)

        elif not self.period_ra and self.score_change:
            print("assigning reward synchronously based on score change!")
            final_reward = self.reward_behavior() * (1 - self.weight) + (self.reward_affect() * self.weight)


        self.cumulative_rl += final_reward
        self.best_rl = np.max([self.best_rl, final_reward])
        self.reset_condition()

        return state, final_reward, d, info

    def handle_level_end(self):
        print("End of level reached, resetting environment.")
        self.reset()
        self.customSideChannel.levelEnd = False
