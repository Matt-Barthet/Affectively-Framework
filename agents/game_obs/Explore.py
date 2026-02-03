import hashlib
import pickle
import random
import copy

import numpy as np
from tensorboardX import SummaryWriter

from affectively.utils.logging import TensorboardGoExplore


hashmap = {0: -100}


def get_state_hash(state):
    state_string = "_".join(str(e) for e in state)
    state_hash = hashlib.md5(state_string.encode()).hexdigest()
    if state_hash not in hashmap:
        hashmap.update({state_hash: len(hashmap)})
    return hashmap[state_hash]


class Cell:

    def __init__(self, state):

        self.key = get_state_hash(state)
        self.trajectory_dict = {"state_trajectory": [],
                                "behavior_trajectory": [],
                                "arousal_trajectory": [],
                                "uncertainty_trajectory": [],
                                "arousal_vectors": [],
                                "score_trajectory": []}

        self.human_vector = []

        self.score, self.previous_score = 0, 0
        self.cumulative_score = 0
        self.arousal = 0
        self.arousal_values = []
        self.uncertainty = 0

        self.arousal_reward = 0
        self.behavior_reward = 0
        self.reward = 0
        self.estimated_position = [0, 0, 0]
        self.age = 0
        self.visited = 1
        self.final = False
        self.policy="Explore"

    def get_cell_length(self):
        return len(self.trajectory_dict['state_trajectory'])

    def update_key(self):
        self.key = get_state_hash(self.trajectory_dict['state_trajectory'][-1])


class Explorer:

    def __init__(self, env, policy="Explore", device="cpu"):
        self.gymnasium_env = env
        self.env = env.env  
        self.policy = policy
        self.archive = {}
        self.updates = 0
        self.bestCell = None
        self.current_cell = Cell([0, 0, 0, 'Start'])
        self.env.step((1, 1, -1))
        self.save_digit = self.current_cell.key
        self.num_timesteps = 0
        self.store_cell(self.current_cell)

    def select_cell(self):
        non_final_cells = [(key, cell) for key, cell in self.archive.items() if not cell.final]
        weights = []
        for key, cell in non_final_cells:
            cell_length = cell.get_cell_length() if cell.get_cell_length() > 0 else 1  # Avoid division by zero
            weight = cell.reward + 1 / cell_length
            weights.append(weight)
        _, chosen_cell = random.choices(non_final_cells, k=1)[0]
        return copy.deepcopy(chosen_cell)

    def store_cell(self, cell):
        if cell is None:
            return False
        if self.store_cell_condition(cell):
            self.archive.update({cell.key: copy.deepcopy(cell)})
            self.update_best_cell(cell)
            return True

    def store_cell_condition(self, cell):
        if cell.key not in self.archive.keys():
            return True
        if cell.reward < self.archive[cell.key].reward:
            return False
        if cell.reward > self.archive[cell.key].reward:
            return True
        if cell.get_cell_length() < self.archive[cell.key].get_cell_length():
            self.updates += 1
            return True
        return False

    def update_best_cell(self, cell):
        if self.bestCell is None:
            self.bestCell = copy.deepcopy(cell)
        elif cell.reward > self.bestCell.reward:
            self.bestCell = copy.deepcopy(cell)
        elif cell.reward == self.bestCell.reward and cell.get_cell_length() < self.bestCell.get_cell_length():
            self.bestCell = copy.deepcopy(cell)

    def save(self, name):
        best_reward = 0
        best_cell = None
        for cell in self.archive.values():
            if cell.reward > best_reward:
                best_reward = cell.reward
                best_cell = cell
        pickle.dump(best_cell, open(f'{name}_Best_Cell.pkl', 'wb'))
        pickle.dump(self.archive, open(f'{name}_Archive.pkl', 'wb'))

    def explore_actions(self, explore_length):
        """
        Return to the current cell using a context load.
        Explore a fixed number of random actions from the current cell.
        """
        self.current_cell = self.select_cell()
        if self.current_cell.get_cell_length() >= 600:
            return
        
        self.gymnasium_env.step((np.inf, np.inf, self.current_cell.key))

        self.env.episode_length = len(self.current_cell.trajectory_dict['state_trajectory'])
        # print(self.current_cell.key, self.current_cell.get_cell_length())
        self.env.previous_score = self.current_cell.previous_score
        self.env.current_score = self.current_cell.score
        self.env.cumulative_rl = self.current_cell.reward
        self.env.cumulative_rb = self.current_cell.cumulative_score
        self.env.estimated_position = self.current_cell.estimated_position

        for j in range(explore_length):

            # input()
            action = self.env.sample_weighted_action()

            if self.current_cell.get_cell_length() >= 600:
                break
            if self.store_cell(self.current_cell):
                self.save_digit = -new_cell.key
            else:
                self.save_digit = 0

            state, _, _, _, _ = self.gymnasium_env.step((action[0], action[1], self.save_digit))
            
            self.num_timesteps += 1

            new_cell = copy.deepcopy(self.current_cell)
            new_cell.trajectory_dict['behavior_trajectory'].append(action)
            new_cell.trajectory_dict['state_trajectory'].append(state)
            new_cell.trajectory_dict['score_trajectory'].append(self.env.cumulative_rb)

            new_cell.update_key()

            new_cell.final = new_cell.get_cell_length() >= 600

            new_cell.previous_score = self.env.previous_score
            new_cell.score = self.env.current_score
            new_cell.cumulative_score = self.env.cumulative_rb
            new_cell.reward = self.env.cumulative_rl

            new_cell.estimated_position = self.env.estimated_position
            self.current_cell = new_cell

        print(len(self.archive), self.bestCell.get_cell_length(), self.bestCell.reward, self.updates)

    def learn(self, total_timesteps, callback = None, explore_length=40, reset_num_timesteps=False):
        for _ in range(total_timesteps):
            self.explore_actions(explore_length)
            self.gymnasium_env.callback.on_episode_end()
        return self.archive
