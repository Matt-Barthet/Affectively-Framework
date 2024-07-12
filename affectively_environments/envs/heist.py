from .base import BaseEnvironment
import numpy as np


class HeistEnvironment(BaseEnvironment):

    def __init__(self, id_number, graphics, weight, path, logging=True):

        """ ---- Heist! specific code ---- """
        self.gridWidth = 9
        self.gridHeight = 9
        self.elementSize = 0.5
        self.position_dict = {}
        self.death_applied = False
        self.previous_health = 0

        self.current_ammo = 0
        self.current_health = 0
        self.previous_angle, self.current_angle = 0, 0

        super().__init__(id_number=id_number, graphics=graphics,
                         obs_space={"low": -np.inf, "high": np.inf, "shape": (341,)},
                         path=path,
                         args=["-agentGridWidth", f"{self.gridWidth}", "-agentGridHeight", f"{self.gridHeight}",
                               "-cellSize", f"{self.elementSize}"], capture_fps=15, time_scale=1, weight=weight,
                         game='Heist', logging=logging)

    def calculate_reward(self, state, position, action):

        self.current_reward = self.current_score - self.previous_score

        position = [round(x / 3) * 3 for x in position]
        position = f"{position[0]},{position[1]},{position[2]}"
        if position not in self.position_dict:
            self.position_dict.update({position: 0})
            self.current_reward += 1

        self.current_angle = np.abs(state[-5])

        if self.current_angle == 0:
            self.current_reward -= 1
        elif self.previous_angle > self.current_angle:
            self.current_reward += 0.1
        elif self.previous_angle < self.current_angle:
            self.current_reward += -0.1

        if self.current_angle == 0:
            pass

        elif self.current_angle < 90:
            self.current_reward += (180 - self.current_angle) / 180
        elif self.current_angle < 180:
            self.current_reward += (180 - self.current_angle) / 180
            self.current_reward -= 0.5

        self.previous_angle = self.current_angle
        self.previous_health = self.current_health

        self.current_reward /= 42

    def reset_condition(self):
        self.episode_length += 1
        if self.episode_length > 300 * 4:
            self.episode_length = 0
            self.create_and_send_message("[Cell Name]:Seed")
            self.reset()
        if self.customSideChannel.levelEnd:
            self.handle_level_end()

    def reset(self, **kwargs):
        state = super().reset()
        self.cumulative_reward = 0
        self.previous_score = 0
        self.position_dict.clear()
        return self.construct_state(state[1], np.asarray(state[0]))

    def construct_state(self, state, grid):
        one_hot = self.one_hot_encode(grid, 4)
        flattened_matrix_obs = [vector for sublist in one_hot for item in sublist for vector in item]
        combined_observations = list(flattened_matrix_obs) + list(state[3:])
        return combined_observations

    def step(self, action):
        transformed_action = [
            action[0] * 4,
            action[1] * 2,
            np.round(action[2]+1),
            np.round(action[3]+1),
            np.round(action[4]/2 + 0.5),
        ]

        # print(transformed_action[2])
        state, env_score, arousal, d, info = super().step(transformed_action)
        position = state[1][:3]

        self.current_health = state[1][4]
        self.current_ammo = state[1][5]

        state = self.construct_state(state[1], np.asarray(state[0]))
        self.calculate_reward(state, position, transformed_action)
        self.cumulative_reward += self.current_reward
        self.best_reward = np.max([self.best_reward, self.current_reward])
        self.reset_condition()
        final_reward = self.current_reward * (1 - self.weight) + (arousal * self.weight)
        return state, final_reward, d, info

    def handle_level_end(self):
        print("End of level reached, resetting environment.")
        self.create_and_send_message("[Cell Name]:Seed")
        self.reset()
        self.customSideChannel.levelEnd = False
