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
from helper import Helper

import matplotlib.pyplot as plt

# def extract_plan(actions_data, best_pos, require_replanning):
#     actions = []
#     save_nums = []
#     pos_x = []
#     if best_pos == None:
#         for i in range(10):
#             actions.append(actions_data["move_right"]["action"])
#             save_nums.append(0)
#             pos_x.append(1)
#         return actions, save_nums, pos_x, require_replanning
    
#     current = best_pos
#     while current.parent != None:
#         for i in range(current.repetitions):
#             actions.append(current.action)
#             save_nums.append(current.save_num)
#             pos_x.append(current.pos_x)

#         if current.state[19] > 0 or current.state[37] > 0:
#             require_replanning = True
        
#         current = current.parent

#     actions.reverse()
#     save_nums.reverse()
#     pos_x.reverse()

#     return actions, save_nums, pos_x, require_replanning

def extract_plan(actions_data, best_pos, require_replanning):
    actions = []
    save_nums = []
    pos_x = []
    if best_pos == None:
        for i in range(10):
            actions.append(actions_data["move_right"]["action"])
            save_nums.append(0)
            pos_x.append(1)
        return actions, save_nums, pos_x, require_replanning
    
    current = best_pos
    while current.parent != None:
        for i in range(current.repetitions):
            actions.append(current.action)
            save_nums.append(current.save_num)
            pos_x.append(current.pos_x)

        if current.state[19] > 0 or current.state[37] > 0:
            require_replanning = True
        
        current = current.parent

    actions.reverse()
    save_nums.reverse()
    pos_x.reverse()

    return require_replanning

def start_search(pos_pool, visited_states, env, dist_x, dist_y, damage, death, starting_state, starting_save_num, latest_save_num, starting_repetitions):
    # print("start dist_x: " + str(dist_x))
    # keyboard.wait("space")
    if death > 0 or damage > 0:
        print("death: " + str(death))
        print("damage: " + str(damage))
        # keyboard.wait("space")

    if len(pos_pool) == 0:
        start_pos = AStarNode   (  
                                    env=env,
                                    parent=None,
                                    dist_x=dist_x,
                                    dist_y=dist_y,
                                    damage=damage,
                                    death=death,
                                    repetitions=starting_repetitions,
                                    action=None,
                                    save_num=starting_save_num
                                )
        
        start_pos.initialize_root(starting_state, starting_save_num)
    
        # print("pos_pool: " + str(pos_pool))
        # print("visited_states: " + str(visited_states))
        # pos_pool = []
        # visited_states = []

    
        children, latest_save_num = start_pos.generate_children(env, starting_save_num, latest_save_num)

        for child in children:
            heapq.heappush(pos_pool, (child.calculate_cost(), child))
        
        current_starting_pos_x = start_pos.pos_x

        best_pos = start_pos
        furthest_pos = start_pos
    else:
        current_starting_pos_x = dist_x
        best_pos, pos_pool = pick_best_pos(pos_pool)
        furthest_pos = best_pos

    return best_pos, furthest_pos, current_starting_pos_x, pos_pool, visited_states, latest_save_num

def pick_best_pos(pos_pool):
    best_pos_pool = None
    best_pos_cost = float("inf")

    for i, current_pos_pool in enumerate(pos_pool):
        current_cost = current_pos_pool[1].calculate_cost()
        if current_cost < best_pos_cost:
            best_pos_pool = current_pos_pool
            best_pos_cost = current_cost
            best_index = i
    
    best_pos_pool = pos_pool.pop(best_index)
    best_pos = best_pos_pool[1]
    return best_pos, pos_pool

def visited(x, y, t, visited_states):
    visited_states.append((x, y, t))

    return visited_states

def is_in_visited(x, y, t, visited_states):
    time_diff = 5
    x_diff = 20
    y_diff = 20

    for v in visited_states:
        if abs(v[0] - x) < x_diff and abs(v[1] - y) < y_diff and abs(v[2] - t) < time_diff and t >= v[2]: 
            return True
        
    return False

def search(time_limit_count, actions_data, env, pos_pool, best_pos, furthest_pos, current_starting_pos_x, latest_save_num, visited_states, require_replanning, original_save_num, original_dist_x, original_dist_y, original_damage, original_death, original_score, original_kill_count):
    
    current = best_pos
    current_good = False
    max_right = 20
    search_count = 0

    # print("----<>")
    # print("best_pos.reached_end_count: " + str(best_pos.reached_end_count))
    # print("search_count: " + str(search_count))
    # print("len(pos_pool): " + str(len(pos_pool)))
    # print("best_pos.pos_x: " + str(best_pos.pos_x))
    # print("current_starting_pos_x: " + str(current_starting_pos_x))
    # print("max_right: " + str(max_right))
    # print("current_good: " + str(current_good))
    # print("env.episode_length: " + str(env.episode_length))
    # print("----<>")

    while best_pos.reached_end_count == 0 and search_count <= 500 and time_limit_count <= 1200 and (len(pos_pool) != 0 and (((best_pos.pos_x - current_starting_pos_x) < max_right) or not current_good) and env.episode_length < 600):        
        # if (search_count % 50) == 0:
        #     print("search count: " + str(search_count))

        if (time_limit_count % 50) == 0:
            print("time limit count: " + str(time_limit_count))
            
        current, pos_pool = pick_best_pos(pos_pool)
        
        if current == None:
            return None
        
        current_good = False
        real_remaining_time, latest_save_num = current.simulate_pos(env, latest_save_num, original_save_num, original_dist_x, original_dist_y, original_damage, original_death, original_score, original_kill_count, best_pos.remaining_time_estimated)
            
        # if best_pos.remaining_time_estimated > current.remaining_time_estimated:
        #     print("Remained Better")
        #     print(str(extract_plan(actions_data, current, require_replanning)))
        #     print("") 

        #     if current.damage == 0 and current.death == 0:
        #         print("no damage")
        #     if current_good:
        #         print("current_good")

        #     keyboard.wait("space")

        check_condition = -1

        if is_in_visited(current.pos_x, current.pos_y, current.time_elapsed, visited_states):
            current.penalty += Helper.visited_list_penalty
        
        if real_remaining_time < 0:
            check_condition = 1
            continue
        elif current.damage > 0 or current.death > 0:
            check_condition = 2
            current.penalty += (Helper.visited_list_penalty * 3)
            heapq.heappush(pos_pool, (current.calculate_cost(), current))
        elif not current.is_in_visited_list and is_in_visited(current.pos_x, current.pos_y, current.time_elapsed, visited_states):
            check_condition = 3
            current_good = True
            current.is_in_visited_list = True
            heapq.heappush(pos_pool, (current.calculate_cost(), current))
        else:
            # if current_good:
            #     if current.damage == 0 and current.death == 0:
            #         if best_pos.remaining_time_estimated > current.remaining_time_estimated:
            #             print("Current changed")
            #             print(str(extract_plan(actions_data, current, require_replanning)))
            #             keyboard.wait("space")

            check_condition = 4
            current_good = True
            visited_states = visited(current.pos_x, current.pos_y, current.time_elapsed, visited_states)

            children, latest_save_num = current.generate_children(env, current.save_num, latest_save_num)

            for child in children:
                heapq.heappush(pos_pool, (child.calculate_cost(), child))

        if current_good:
            if current.damage == 0 and current.death == 0:
                if best_pos.remaining_time_estimated > current.remaining_time_estimated or current.reached_end_count > 0:
                    best_pos = current
                    # print("Search Count Reset")
                    search_count = 0
                    # print("Best changed")
                    # print("best_pos.pos_x: " + str(best_pos.pos_x))
                    # # print(str(extract_plan(actions_data, best_pos, require_replanning)))
                    # print("=========") 
                    # keyboard.wait("space")

            if current.pos_x > furthest_pos.pos_x:
                furthest_pos = current

        search_count+=1  
        time_limit_count+=1             

    if (current.pos_x - current_starting_pos_x) < max_right and furthest_pos.pos_x > best_pos.pos_x + 20:
        best_pos = furthest_pos

    # print("best_pos: " + str(best_pos))
    return time_limit_count, search_count, best_pos, furthest_pos, pos_pool, visited_states, latest_save_num

def optimise(time_limit_count, pos_pool, visited_states, actions_data, env, original_state, original_save_num, original_dist_x, original_dist_y, original_damage, original_death, original_score, original_kill_count):
    plan_ahead = 2
    steps_per_search = 1

    require_replanning = False
    latest_save_num = original_save_num

    state = original_state

    best_pos, furthest_pos, current_starting_pos_x, pos_pool, visited_states, latest_save_num = start_search(pos_pool, visited_states, env, original_dist_x, original_dist_y, original_damage, original_death, state, latest_save_num, latest_save_num, steps_per_search)

    if state[37] > 0:
        best_pos, furthest_pos, current_starting_pos_x, pos_pool, visited_states, latest_save_num = start_search(pos_pool, visited_states, env, original_dist_x, original_dist_y, original_damage, original_death, original_state, original_save_num, latest_save_num, steps_per_search)

    time_limit_count, search_count, best_pos, furthest_pos, pos_pool, visited_states, latest_save_num = search(time_limit_count, actions_data, env, pos_pool, best_pos, furthest_pos, current_starting_pos_x, latest_save_num, visited_states, require_replanning, original_save_num, original_dist_x, original_dist_y, original_damage, original_death, original_score, original_kill_count)
    
    # require_replanning = extract_plan(actions_data, best_pos, require_replanning)

    return time_limit_count, search_count, best_pos, latest_save_num, pos_pool, visited_states

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
        
        main_save_load_num = 0

        main_save_load_num += 1
        main_env.step(main_actions_data["stay_still"]["action"], -main_save_load_num)

        main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], main_save_load_num)
        
        main_dist_x = 0
        main_dist_y = 0
        main_damage = 0
        main_death = 0
        main_score = 0
        main_kill_count = 0
        main_time_limit_count = 0

        main_pos_pool = []
        main_visited_states = []

        while True:   
            main_new_save_load_num = main_save_load_num

            # print("LOAD TO START")
            main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], main_save_load_num)
            # keyboard.wait("space")

            main_time_limit_count, main_search_count, main_best_pos, main_new_save_load_num, main_pos_pool, main_visited_states = optimise(main_time_limit_count, main_pos_pool, main_visited_states, main_actions_data, main_env, main_state, main_new_save_load_num, main_dist_x, main_dist_y, main_damage, main_death, main_score, main_kill_count)

            # print("main_search_count: " + str(main_search_count))

            if main_search_count >= 500 or main_search_count == 0 or main_time_limit_count >= 1200:
                playable = False
                print("Close 1")
                main_env.env.close()
                return playable, main_dist_x
            
            # print("MOVE")
            # print("LOAD TO START")
            main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], main_save_load_num)
            # keyboard.wait("space")

            if main_best_pos == None:
                test = 1
                # # print("1 LOAD MOVE: Stay Still")
                # main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], main_save_load_num)
                # # keyboard.wait("space")

                # main_dist_x += main_state[0]
                # # print("1 Dist X: " + str(main_dist_x))
                # main_dist_y += main_state[1]
                # main_damage += main_state[19]
                # main_death += main_state[37]

                # main_score = main_state[7]
                # main_kill_count = main_state[23]

                # # print("1 SAVE")
                # main_new_save_load_num += 1
                # main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], -main_new_save_load_num)
                # # keyboard.wait("space")

                # if main_reached_termination or main_death > 0:
                #     # print("TERMINATION 1")
                #     # print("main_reached_termination: " + str(main_reached_termination))
                #     # print("main_death: " + str(main_death))

                #     # if main_death > 0:
                #     #     print("TERMINATION 1")
                #     #     keyboard.wait("space")
                        
                #     playable = True

                #     if main_reached_end_door == False:
                #         playable = False

                #     print("Close 2")
                #     main_env.env.close()

                #     return playable, main_dist_x
            else:
                main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_best_pos.action, main_best_pos.save_num)
                
                main_dist_x = main_best_pos.pos_x
                main_dist_y = main_best_pos.pos_y
                main_damage += main_best_pos.damage
                main_death += main_best_pos.death

                main_score = main_best_pos.score_difference
                main_kill_count = main_best_pos.kill_count_difference

                # keyboard.wait("space")   

                if main_reached_termination or main_death > 0:
                    # print("TERMINATION 2")
                    # print("main_reached_termination: " + str(main_reached_termination))
                    # print("main_death: " + str(main_death))

                    # if main_death > 0:
                    #     print("TERMINATION 2")
                    #     keyboard.wait("space")

                    playable = True

                    if main_reached_end_door == False:
                        playable = False

                    print("Close 3")
                    main_env.env.close()

                    return playable, main_dist_x 

            # print("SAVE")
            main_new_save_load_num += 1  
            main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], -main_new_save_load_num)
            # keyboard.wait("space")
            
            if main_reached_termination or main_death > 0:
                # print("TERMINATION 3")
                # print("main_reached_termination: " + str(main_reached_termination))
                # print("main_death: " + str(main_death))

                # if main_death > 0:
                #     print("TERMINATION 3")
                #     keyboard.wait("space")

                playable = True

                if main_reached_end_door == False:
                    playable = False

                print("Close 4")
                main_env.env.close()

                return playable, main_dist_x
            
            # print("LOAD NEW MOVE")
            main_save_load_num = main_new_save_load_num
            test_raw_grid, test_state, test_reached_termination, test_reached_end_door, test_reward, test_done, test_info = main_env.step(main_actions_data["stay_still"]["action"], main_save_load_num) 
            # keyboard.wait("space") 
            
            if main_reached_termination and main_death > 0:
                # if main_death > 0:
                #     print("TERMINATION 4")
                #     keyboard.wait("space")

                playable = True

                if main_reached_end_door == False:
                    playable = False

                print("Close 5")
                main_env.env.close()

                return playable, main_dist_x
            
            print("Distance Travelled: " + str(main_dist_x))

            if main_reached_termination and main_death > 0:
                playable = True

                if main_reached_end_door == False:
                    playable = False

                print("Close 6")
                main_env.env.close()

                return playable, main_dist_x
            
            print("MOVED")
            print("******************************************")
            # keyboard.wait("space")
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

# cd C:\Users\vassa\Documents\GitHub\Affectively-Framework\Tensorboard\CNN
# tensorboard --logdir=C:\Users\vassa\Documents\GitHub\Affectively-Framework\Tensorboard\CNN
# 

# cd C:\Users\vassa\Documents\GitHub\Affectively-Framework
# conda activate unity_gym
# python AstarAgent.py
# 

# cd C:\Users\vassa\Documents\GitHub\Affectively-Framework
# conda activate unity_gym
# python GANMain.py
# 

if __name__ == "__main__":  
    test = 1
    # env = GANLevelEnv()
    # model = PPO.load("GANArousalAgents/PPO/cnn_ppo_solid_optimize_1_extended.zip", env=env)
    # obs, info = env.reset()

    # segment_buffer = deque(maxlen=11)

    # actions = []
    # for i in range(11):
    #     action, _ = model.predict(obs, deterministic=True)
    #     actions.append(action.copy())
    #     print("action: " + str(action))
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     # keyboard.wait("space")

    #     segment = info["segment"]
    #     segment_buffer.append(segment)
    
    # actions = np.array(actions)

    # num_steps, action_dim = actions.shape

    # x = np.tile(np.arange(action_dim), num_steps)
    # y = actions.flatten()

    # plt.figure(figsize=(10, 5))
    # plt.scatter(x, y, alpha=0.6)
    # plt.xlabel("Action dimension index")
    # plt.ylabel("Action value")
    # plt.title("Scatter plot of all action values")
    # plt.grid(True)
    # plt.show()
    # # agent = AstarAgent()
    # # agent.AStarRun(segment_buffer) 