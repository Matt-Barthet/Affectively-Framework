import hashlib
import pickle
import random
import copy

import numpy as np
from tensorboardX import SummaryWriter

from affectively_environments.utils.logging import TensorboardGoExplore


def get_state_hash(state):
    state_string = "_".join(str(e) for e in state)
    state_hash = hashlib.md5(state_string.encode()).hexdigest()
    return state_hash


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

        self.score = 0
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

    def get_cell_length(self):
        return len(self.trajectory_dict['state_trajectory'])

    def update_key(self):
        self.key = get_state_hash(self.trajectory_dict['state_trajectory'][-1])


class Explorer:

    def __init__(self, env, name):
        self.name = name
        self.env = env
        self.archive = {}

        self.writer = SummaryWriter(log_dir=f'./Tensorboard/Go-Explore/{name}')
        self.callback = TensorboardGoExplore(self, self)

        self.env.create_and_send_message(f"[Save Dir]:./Results/Go-Explore")
        self.env.create_and_send_message(f"[Save Name]:{name}")

        self.updates = 0

        self.current_cell = Cell([0, 0, 0, 'Start'])
        self.archive = {self.current_cell.key: self.current_cell}
        self.env.create_and_send_message(f"[Save]:{self.current_cell.key}")
        self.bestCell = self.current_cell

    def select_cell(self):
        print("Selecting cell...", self.bestCell.get_cell_length(), self.bestCell.reward)
        non_final_cells = [(key, cell) for key, cell in self.archive.items() if not cell.final]

        weights = []
        for key, cell in non_final_cells:
            cell_length = cell.get_cell_length() if cell.get_cell_length() > 0 else 1  # Avoid division by zero
            weight = cell.reward + 1 / cell_length
            weights.append(weight)
        chosen_key, chosen_cell = random.choices(non_final_cells, weights=weights, k=1)[0]

        self.env.episode_length = len(self.current_cell.trajectory_dict['state_trajectory'])
        self.env.previous_score = chosen_cell.score
        self.env.current_score = chosen_cell.score
        self.env.cumulative_reward = chosen_cell.reward
        self.env.estimated_position = chosen_cell.estimated_position
        print("Selected cell...", self.bestCell.get_cell_length(), self.bestCell.reward)

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

    def save_best_cells(self, name):
        best_reward = 0
        best_cell = None
        for cell in self.archive.values():
            if cell.reward > best_reward:
                best_reward = cell.reward
                best_cell = cell
        pickle.dump(best_cell, open(f'./Results/Go-Explore/{name}_Best_Cell.pkl', 'wb'))
        pickle.dump(self.archive, open(f'./Results/Go-Explore/{name}_Archive.pkl', 'wb'))

    def create_cell(self, action, state):
        self.current_cell.trajectory_dict['behavior_trajectory'].append(action)
        self.current_cell.trajectory_dict['state_trajectory'].append(state)
        # self.current_cell.trajectory_dict['arousal_vectors'].append(arousal_vector)
        self.current_cell.trajectory_dict['score_trajectory'].append(self.env.current_score)
        self.current_cell.update_key()

        self.current_cell.final = self.current_cell.get_cell_length() >= 600
        self.current_cell.reward = self.env.cumulative_reward
        self.current_cell.score = self.env.current_score
        self.current_cell.estimated_position = self.env.estimated_position

    def explore_actions(self, explore_length):
        """
        Return to the current cell using a context load.
        Explore a fixed number of random actions from the current cell.
        """
        self.current_cell = self.select_cell()
        self.env.create_and_send_message(f"[Cell Name]:{self.current_cell.key}")
        self.env.reset()

        for j in range(explore_length):
            action = self.env.action_space.sample()
            state, score, d, info = self.env.step((action[0], action[1]))
            self.create_cell(action, state)
            if self.current_cell.get_cell_length() >= 600:
                break
            if self.store_cell(self.current_cell):
                self.env.create_and_send_message(f"[Save States]:{self.current_cell.key}")

        print(len(self.archive), self.bestCell.get_cell_length(), self.bestCell.reward)

    def explore(self, name, total_rollouts, explore_length):
        for x in range(total_rollouts):
            self.explore_actions(explore_length)
            self.env.callback.on_episode_end()
            self.callback.on_step()
        self.save_best_cells(name)
        self.writer.close()
