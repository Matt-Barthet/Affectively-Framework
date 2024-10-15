from abc import ABC
import configparser

import mlagents_envs.exception
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from BaseEnvironment import BaseEnvironment
from GoBlend.Archive import Archive
from GoBlend.Cell import Cell
from GoBlend.RewardFunctions import reward_functions
from Utils.Tensorboard_Callbacks import TensorboardGoExplore


class GoBlendEnvironment(BaseEnvironment, ABC):

    def __init__(self, id_number, graphics, path, config):

        """ ---- Pirates! specific code ---- """
        self.gridWidth = 5
        self.gridHeight = 5
        self.elementSize = 1
        # State = direction, velocity x+y, health, powerup, grid

        obs_space = {"low": -np.inf, "high": np.inf, "shape": (1, )}
        super().__init__(id_number, graphics, obs_space, path, ["-gridWidth", f"{self.gridWidth}", "-gridHeight", f"{self.gridHeight}", "-elementSize", f"{self.elementSize}"])

        """ ---- Generic GoBlend code ---- """
        self.config = config  # Go-Explore configuration file for this experiment
        self.archive = Archive(config)  # Go-Explore's archive of cells.

        self.total_timesteps = int(config['Cells']['max_trajectories'])  # total number of "rollouts" to explore
        self.explore_length = int(config['Cells']['explore_steps'])  # length of each rollout during exploration
        self.max_trajectory_length = int(config['Cells']['max_trajectory_size'])

        self.lambdaValue = float(config['Rewards']['lambda'])
        self.behavior_target = config['Rewards']['behavior_target']
        self.behavior_function = reward_functions[config['Rewards']['behavior_reward']]
        self.normalize_behavior = config['Rewards']['normalize_behavior']
        self.arousal_target = config['Rewards']['arousal_target']
        self.arousal_function = reward_functions[config['Rewards']['arousal_reward']]
        self.kNN = int(config['Human Model']['kNN'])

        self.writer = SummaryWriter(log_dir='./Tensorboard/GoBlend2')  # Logger for tensorboard
        self.callback = TensorboardGoExplore(self, self.archive)  # Callback class for passing stats to tensorboard

        self.create_and_send_message(f"[Grid]:{self.gridWidth},{self.gridHeight},{self.elementSize}")
        self.create_and_send_message(f"[CWD]:../GoBlend/Test_Run/")

        new_state = []  # Convert raw state into cell representation
        arousal = []  # Get arousal value from side channel

        self.current_cell = None
        self.create_cell((1, 0), new_state, arousal, 0)
        if self.archive.store_cell(self.current_cell):
            self.create_and_send_message(f"[Save]:{self.current_cell.key}")

    def reset(self, **kwargs):
        state = self.env.reset()
        return state

    def create_cell(self, action, state, arousal_vector, score):
        if self.current_cell is not None:
            self.current_cell.trajectory_dict['behavior_trajectory'].append(action)
            self.current_cell.trajectory_dict['state_trajectory'].append(state)
            self.current_cell.trajectory_dict['arousal_vectors'].append(arousal_vector)
            self.current_cell.trajectory_dict['score_trajectory'].append(score)
            self.current_cell.update_key(state)
        else:
            self.current_cell = Cell(state, {"state_trajectory": [state],
                                             "behavior_trajectory": [action],
                                             "arousal_trajectory": [],
                                             "uncertainty_trajectory": [],
                                             "arousal_vectors": [arousal_vector],
                                             "score_trajectory": [score]})

        self.current_cell.assess_cell(self.lambdaValue, self.normalize_behavior == "True",
                                      self.arousal_function)

    def construct_state(self, vector, visual):
        vector[0] = (vector[0] // 30) * 30
        vector = [vector[0]] + list(vector[4:])
        visual_flat = [element[0] for row in visual for element in row]
        for element in range(len(visual_flat)):
            if visual_flat[element] == 7 or visual_flat[element] == 4:
                visual_flat[element] = 0
            elif visual_flat[element] == 2:
                visual_flat[element] = 1
            elif visual_flat[element] == 3:
                visual_flat[element] = 2
            else:
                visual_flat[element] = 3
        new_state = list(vector) + visual_flat
        return new_state

    def step(self, action):
        # Move the env forward 1 tick and receive messages through side-channel.
        state, env_score, d, info = self.env.step((action[0] - 1, action[1]))
        new_state = self.construct_state(state[1], state[0])
        ended = self.customSideChannel.levelEnd
        return new_state, env_score, ended

    def explore_actions(self):
        """
        Return to the current cell using a context load.
        Explore a fixed number of random actions from the current cell.
        """
        self.current_cell = self.archive.select_cell()  # Select a cell from the archive with the config's strategy
        # Setting the size of the 2D grid used for go-explore
        self.create_and_send_message(f"[Cell Name]:{self.current_cell.key}")  # Perform a context load with the given ID
        for j in range(self.explore_length):
            action = self.env.action_space.sample()
            state, score, ended = self.step((action[0], action[1]))
            arousal_vector = None  # TODO - not now
            self.create_cell(action, state, arousal_vector, score)
            if self.current_cell.get_cell_length() >= self.max_trajectory_length:
                break
            if self.archive.store_cell(self.current_cell):
                self.create_and_send_message(f"[Save]:{self.current_cell.key}")
            if ended:
                break

    def explore(self):
        for _ in range(self.total_timesteps):
            self.explore_actions()  # Perform a rollout of random actions
            self.callback.on_step()  # Update tensorboard after each rollout
        self.writer.close()


if __name__ == "__main__":
    config_reader = configparser.ConfigParser()
    config_reader.read('./GoBlend/config_files/baseline.config')
    env = GoBlendEnvironment(0,
                             graphics=True,
                             path="./Builds/Pirates/Platform.exe",
                             config=config_reader)
    env.explore()  # Run the exploration phase of go-explore
