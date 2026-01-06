# fro÷ßm sb3_contrib import RecurrentPPO
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import argparse

from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.utils.logging import TensorBoardCallback
from agents.game_obs.Rainbow_DQN import RainbowAgent
import torch

import subprocess
import sys

import time
import random
from collections import deque

# from stable_baselines3 import PPO
# from GANEnv import GANLevelEnv
# from CNet.model import CNet

# Install pathfinding if it's not already installed
try:
    import pathfinding
    import keyboard
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pathfinding"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard"])
    import pathfinding
    import keyboard

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.core.node import Node
from pathfinding.finder.a_star import AStarFinder
import numpy as np

import heapq
from node import AStarNode

def heuristic(state, reward):
    dead = state[37]
    damaged = state[19]

    if dead:      # dead
        return -9999, dead, damaged
    forward = state[0]
    return (forward + reward), dead, damaged

def simulate_action(save_num_list, env, action, save_load_num):
    raw_grid, state, reached_termination, reached_end_door, reward, done, info = env.step(action, save_load_num)
    keyboard.wait("space")
    score, dead, damaged = heuristic(state, reward)
    return score, done, dead, damaged, reached_termination, reached_end_door

def extract_plan(node):
    action_plan = []
    save_plan = []
    while node.parent is not None:
        action_plan.append(node.action)
        save_plan.append(node.save_id)
        node = node.parent
    action_plan.reverse()
    save_plan.reverse()
    return action_plan, save_plan

def astar_search(save_num_list, env, selected_save_id, actions_data, max_depth):
    open_set = []
    visited = set()

    env.step(actions_data["stay_still"]["action"], -selected_save_id)
    keyboard.wait("space")
    save_num_list.append(selected_save_id)

    root = AStarNode(parent=None,
                action_name=None,
                action=None,
                depth=0,
                score=0,
                dead=0,
                damaged=0,
                save_id=selected_save_id)
    
    heapq.heappush(open_set, root)

    best_node = root

    MAX_EXPANSIONS = 40
    expansions = 0

    while open_set:
        expansions += 1

        node = heapq.heappop(open_set)
        
        if node.depth >= max_depth:
            continue        

        for name, entry in actions_data.items():
            selected_save_id += 1
            score, done, dead, damaged, reached_termination, reached_end_door = simulate_action(save_num_list, env, entry["action"], node.save_id)

            child = AStarNode(parent=node,
                        action_name=name,
                        action=entry["action"],
                        depth=node.depth + 1,
                        score=node.score + score,
                        dead=node.dead + dead,
                        damaged=node.damaged + damaged,
                        save_id=selected_save_id
                        )

            if child.score > best_node.score:
                score, done, dead, damaged, reached_termination, reached_end_door = simulate_action(save_num_list, env, entry["action"], node.save_id)
                best_node = child

            if not done:
                heapq.heappush(open_set, child)

            env.step(actions_data["stay_still"]["action"], -selected_save_id)
            keyboard.wait("space")
            save_num_list.append(selected_save_id)

    action_plan, save_plan = extract_plan(best_node)

    return best_node, action_plan, save_plan, selected_save_id, save_num_list

class AstarAgent:    
    # cwd: c:\Users\vassa\Documents\GitHub\Affectively-Framework
    # conda_env: affect-envs
    # script_path: ./train.py
    # runs: 5
    # use_gpu: 0
    # weight: 0.5
    # cluster: 0
    # target_arousal: 1
    # preference: 1
    # classifier: 1
    # game: platform
    # period_ra: 0
    # cv: 0
    # headless: 1
    # discretize: 0
    # grayscale: 0
    # output_dir: ./results/
    # algorithm: PPO
    # policy: MlpPolicy

    def AStarRun(self, segments) :
        weight = 0.5
        target_arousal = 1
        cluster = 0
        period_ra = 0
        headless = 1
        grayscale = 0
        discretize = 0
        use_gpu = 0
        classifier = 1
        preference = 1

        main_env = PiratesEnvironmentGameObs(
            id_number=1,
            weight=weight,
            graphics=True, # Pirates is bugged in headless, prevent it manually for now
            cluster=cluster,
            target_arousal=target_arousal,
            period_ra=period_ra,
            discretize=discretize,
            classifier=classifier,
            preference=preference,
        )

        main_env.build_segment(segments)
        
        
        main_actions_data = {
            "stay_still": {"action": (1, 0, 0), "score": 0},
            "move_left": {"action": (0, 0, 0), "score": 0},
            "move_right": {"action": (2, 0, 0), "score": 0},
            "jump_straight": {"action": (1, 1, 0), "score": 0},
            "jump_left": {"action": (0, 1, 0), "score": 0},
            "jump_right": {"action": (2, 1, 0), "score": 0},
        }

        self.obs = main_env.reset()

        # ganenv = GANLevelEnv()
        # model = PPO.load("PPO/cnn_ppo_solid_optimize_1_extended.zip", env=ganenv)
        # obs, info = ganenv.reset()
        # action, _ = model.predict(obs, deterministic=True)
        # obs, reward, terminated, truncated, info = ganenv.step(action)
        # keyboard.wait("space")
        
        main_save_load_num = 1
        main_save_num_list = []

        testing_counting = 0
        moving_counting = 0

        testing_counting += 1
        moving_counting += 1
        print(f"{moving_counting}: {testing_counting}: SAVE")
        main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["jump_right"]["action"], -testing_counting)
        keyboard.wait("space")

        # main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], -main_save_load_num)
        # keyboard.wait("space")

        main_save_num_list.append(main_save_load_num)

        temprows, tempcols, t = main_raw_grid.shape
        main_player_pos = ((int((temprows - 1) / 2)), (int((tempcols - 1) / 2)))

        testCount = 0            
        
        main_pos_x = 0

        while True:   
            moving_counting += 1

            # if moving_counting % 2 == 0:
            #     print(f"{moving_counting}: {testing_counting}: LOAD")
            #     print("ACTION 1: " + str(main_actions_data["jump_right"]["action"]))
            #     main_env.step(main_actions_data["jump_right"]["action"], testing_counting)
            #     keyboard.wait("space")
            # else:
            #     testing_counting += 1
            #     print(f"{moving_counting}: {testing_counting}: SAVE")
            #     print("ACTION 1: " + str(main_actions_data["jump_right"]["action"]))
            #     main_env.step(main_actions_data["jump_right"]["action"], -testing_counting)
            #     keyboard.wait("space")

            # keyboard.wait("space")
            
            main_new_save_load_num = main_save_load_num
            main_best_node, main_action_plan, main_save_plan, main_new_save_load_num, main_save_num_list = astar_search(main_save_num_list, main_env, main_save_load_num, main_actions_data, max_depth=3)

            if main_best_node.action_name != None:
                print(f"main_action_plan: {main_action_plan}")
                print(f"main_save_plan: {main_save_plan}")
                print(f"main_best_node.action_name: {main_best_node.action_name}")
                print(f"main_best_node.parent: {main_best_node.parent}")
                print(f"main_best_node.parent.save_id: {main_best_node.parent.save_id}")
                score, done, dead, damaged, main_reached_termination, main_reached_end_door = simulate_action(main_save_num_list, main_env, main_best_node.action, main_best_node.parent.save_id)
            else:
                score, done, dead, damaged, main_reached_termination, main_reached_end_door = simulate_action(main_save_num_list, main_env, main_actions_data["stay_still"]["action"], main_save_load_num)

            if main_reached_termination:
                playable = True

                if main_reached_end_door == False:
                    playable = False

                main_env.env.close()
                return playable
            
            main_save_load_num = main_new_save_load_num + 1

        # 0 - (transform.position - previousPosition).x
        # 1 - (transform.position - previousPosition).y
        # 2 - _corgiController.Speed.x
        # 3 - _corgiController.Speed.y
        # 4 - _corgiController._movementDirection
        # 5 - _health.CurrentHealth
        # 6 - _health.hasPowerUp
        # 7  - Player Score
        # 8 - Player Has Collisions
        # 9 - Player Is Colliding Above
        # 10 - Player Is Colliding Below
        # 11 - Player Is Colliding Left
        # 12 - Player Is Colliding Right
        # 13 - Player Is Falling
        # 14 - Player Is Grounded
        # 15 - Player Is Jumping
        # 16 - Player Speed X
        # 17 - Player Speed Y
        # 18 - Player Health
        # 19 - Player Damaged
        # 20 - Player Point Pickup
        # 21 - Player Power Pickup
        # 22 - Player Has Powerup
        # 23 - Player Kill Count
        # 24 - Bots Visible
        # 25 - Bot Has Collisions
        # 26 - Bot Is Colliding Below
        # 27 - Bot Is Colliding Left
        # 28 - Bot Is Colliding Right
        # 29 - Bot Is Falling
        # 30 - Bot Is Grounded
        # 31 - Bot Speed X
        # 32 - Bot Speed Y
        # 33 - Bot Health
        # 34 - Bot Player Distance
        # 35 - Pick Ups Visible
        # 36 - Pick Up Player Disctance
        # 37 - Player Death


# cd C:\Users\vassa\Documents\GitHub\Affectively-Framework
# conda activate unity_gym
# python PPOAgent.py
# 


if __name__ == "__main__":  
    test = 1
    # env = GANLevelEnv()
    # model = PPO.load("GANArousalAgents/PPO/cnn_ppo_solid_optimize_1_extended.zip", env=env)
    # obs, info = env.reset()

    # segment_buffer = deque(maxlen=11)

    # for i in range(11):
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     keyboard.wait("space")

    #     segment = info["segment"]
    #     segment_buffer.append(segment)
    
    # agent = AstarAgent()
    # agent.AStarRun(segment_buffer) 