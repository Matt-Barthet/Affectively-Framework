import copy
import pickle
import numpy as np
import random


class Archive:

    def __init__(self, config):

        self.selection_method = config['Cells']['cell_selection']  # Algorithm for selecting cells for exploration
        self.cellSelectionFunctions = {"Random": self.select_random_cell, "Roulette": self.select_cell_roulette}
        self.explore_steps = int(config['Cells']['explore_steps'])  # Number of actions to explore per iteration
        self.selectionLambda = config['Cells']['cell_selection_lambda']  # How to blend the reward for selection

        self.epsilon = float(config['Rewards']['epsilon'])  # Minimum improvement required for storing the cell
        self.behavior_target = config['Rewards']['behavior_target']  # Our reward type for behavior
        self.arousal_target = config['Rewards']['arousal_target']  # Our reward type for affect

        self.archive = {}
        self.bestCell = None

    def select_cell(self):
        return self.cellSelectionFunctions[self.selection_method]()

    def select_random_cell(self):
        key, cell = random.choice(list(self.archive.items()))
        if cell.final:
            return self.select_random_cell()
        else:
            return copy.deepcopy(cell)

    def select_cell_roulette(self):
        weights = [cell.assess_cell(self.selectionLambda, self.behavior_target == "Imitate") for cell in self.archive]
        weights = np.asarray(weights) / np.sum(weights)
        cell = np.random.choice(list(self.archive.items()), size=1, replace=False, p=weights)
        if cell.final:
            self.select_cell_roulette()
        return copy.deepcopy(cell)

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
        if cell.blended_reward < self.archive[cell.key].blended_reward:
            return False
        if cell.blended_reward >= self.archive[cell.key].blended_reward + self.epsilon:
            return True
        return cell.get_cell_length() < self.archive[cell.key].get_cell_length()

    def update_best_cell(self, cell):
        if cell.get_cell_length() == 10 and (self.bestCell is None or cell.blended_reward > self.bestCell.blended_reward):
            self.bestCell = copy.deepcopy(cell)

    def save_best_cells(self):
        best_reward = 0
        best_cell = None
        for cell in self.archive.values():
            if cell.blended_reward > best_reward:
                best_reward = cell.blended_reward
                best_cell = cell
        pickle.dump(best_cell, open('./Best_Cell.pkl', 'wb'))
        pickle.dump(self.archive, open('./Archive.pkl', 'wb'))
