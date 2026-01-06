from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from mlagents_envs.exception import UnityTimeOutException
from tqdm import tqdm

import argparse
import traceback
import time

from affectively.environments.pirates_cv import PiratesEnvironmentCV
from affectively.environments.heist_cv import HeistEnvironmentCV
from affectively.environments.solid_cv import SolidEnvironmentCV
from affectively.environments.heist_game_obs import HeistEnvironmentGameObs
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.utils.logging import TensorBoardCallback
from agents.game_obs.Rainbow_DQN import RainbowAgent
import torch

import keyboard
import numpy as np

from node import Node
import heapq

class PersistentProgressBarCallback(BaseCallback):

    def __init__(self, total_timesteps, env_wrapper, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.env_wrapper = env_wrapper
        self.pbar = None
        self.last_update_step = 0

    def _on_training_start(self):
        if self.pbar is not None:
            try:
                self.pbar.close()
            except:
                pass

        self.pbar = tqdm(
            total=self.total_timesteps,
            initial=self.model.num_timesteps,
            desc="Training",
            unit=" steps"
        )
        self.last_update_step = self.model.num_timesteps

    def _on_step(self):
        if self.pbar is not None:
            steps_since_last = self.model.num_timesteps - self.last_update_step
            if steps_since_last > 0:
                self.pbar.update(steps_since_last)
                self.last_update_step = self.model.num_timesteps

            if hasattr(self.env_wrapper, 'callback') and self.env_wrapper.callback is not None:
                callback = self.env_wrapper.callback
                postfix = {
                    'Best Score': f"{callback.best_env_score:.1f}",
                    'Best R_a': f"{callback.best_cumulative_ra:.2f}",
                    'Best R_b': f"{callback.best_cumulative_rb:.2f}",
                    'Episodes': callback.episode
                }
                self.pbar.set_postfix(postfix)

        return True

    def _on_training_end(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


def create_environment(args, run):
    if args.cv == 0:
        if args.game == "fps":
            return HeistEnvironmentGameObs(
                id_number=run,
                weight=args.weight,
                graphics=args.headless == 0,
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra,
                discretize=args.discretize,
                classifier=args.classifier,
                preference=args.preference,
                decision_period=args.decision_period,
            )
        elif args.game == "solid":
            return SolidEnvironmentGameObs(
                id_number=run,
                weight=args.weight,
                graphics=args.headless == 0,
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra,
                discretize=args.discretize,
                classifier=args.classifier,
                preference=args.preference,
                decision_period=args.decision_period,
            )
        elif args.game == "platform":
            return PiratesEnvironmentGameObs(
                id_number=run,
                weight=args.weight,
                graphics=True,
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra,
                discretize=args.discretize,
                classifier=args.classifier,
                preference=args.preference,
                decision_period=args.decision_period,
            )
    elif args.cv == 1:
        if args.game == "fps":
            return HeistEnvironmentCV(
                id_number=run,
                weight=args.weight,
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra,
                grayscale=args.grayscale,
                classifier=args.classifier,
                preference=args.preference,
                decision_period=args.decision_period,
            )
        elif args.game == "solid":
            return SolidEnvironmentCV(
                id_number=run,
                weight=args.weight,
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra,
                grayscale=args.grayscale,
                classifier=args.classifier,
                preference=args.preference,
                decision_period=args.decision_period,
            )
        elif args.game == "platform":
            return PiratesEnvironmentCV(
                id_number=run,
                weight=args.weight,
                cluster=args.cluster,
                target_arousal=args.target_arousal,
                period_ra=args.periodic_ra,
                grayscale=args.grayscale,
                classifier=args.classifier,
                preference=args.preference,
                decision_period=args.decision_period,
            )
    return None


def close_environment_safely(env):
    try:
        if hasattr(env, 'env'):
            env.env.close()
        else:
            env.close()
        print("✓ Environment closed")
    except Exception as e:
        print(f"Warning during env close: {e}")
    print("⏳ Waiting for ports to be released...")
    time.sleep(3)


def close_callback_safely(callback):
    if callback is None:
        return

    try:
        if hasattr(callback, 'writer') and callback.writer is not None:
            callback.writer.close()
            print("✓ TensorBoard writer closed")
    except Exception as e:
        print(f"Warning closing callback writer: {e}")


def close_progress_bar_safely(callbacks):
    if callbacks is None:
        return

    try:
        if hasattr(callbacks, 'pbar') and callbacks.pbar is not None:
            callbacks.pbar.close()
            print("✓ Progress bar closed")
    except Exception as e:
        print(f"Warning closing progress bar: {e}")


def train_with_recovery(model, env, callbacks, total_timesteps, max_retries=5):
    retry_count = 0

    while retry_count < max_retries:
        try:
            print(f"📊 Starting/resuming training at timestep: {model.num_timesteps}/{total_timesteps}")

            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                reset_num_timesteps=False
            )

            print(f"✅ Training completed successfully! Final timesteps: {model.num_timesteps}")
            close_progress_bar_safely(callbacks)

            return True

        except UnityTimeOutException as e:
            retry_count += 1
            print(f"\n⚠️ Unity timeout at timestep {model.num_timesteps} (attempt {retry_count}/{max_retries})")
            print(f"Error: {e}")

            close_progress_bar_safely(callbacks)

            if retry_count < max_retries:
                print("🔄 Attempting to recover...")
                return False
            else:
                print(f"❌ Max retries ({max_retries}) reached at timestep {model.num_timesteps}")
                raise

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            traceback.print_exc()
            close_progress_bar_safely(callbacks)
            raise
    return None

def calculate_score(state, score):
    direction = state[0]
    isDead = state[37]

    if isDead:
        return -999
    else:
        return (direction + score)

def heuristic(state, reward):
    dead = state[37]
    damaged = state[19]

    if dead:      # dead
        return -9999, dead, damaged
    forward = state[0]
    return (forward + reward), dead, damaged

def check_save_num(env, save_load_num, save_num_list):
    if save_load_num in save_num_list:
        print("Save number Exists: " + str(save_load_num))
    else:
        print("Save number DOES NOT Exist: " + str(save_load_num))
        # keyboard.wait("space")

    if env.episode_length > 100:
        print("Time is greater than 100: " + str(env.episode_length))
    else:
        print("Time is less than 100: " + str(env.episode_length))
        # keyboard.wait("space")

def simulate_action(save_num_list, env, action, save_load_num):
    raw_grid, state, reward, done, info = env.step(action, save_load_num)
    # print("SIMULATION")
    # check_save_num(env, save_load_num, save_num_list)

    # print("simulate action: " + str(action))
    # keyboard.wait("space")
    score, dead, damaged = heuristic(state, reward)
        
    return score, done, dead, damaged


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

def print_action_path(actions_data, save_num_list, env, node, show_steps):
    action_names = []
    actions = []
    load_nums = []
    current = node

    dead_count = 0
    damaged_count = 0

    while current is not None:
        if current.action_name is not None:
            action_names.append(current.action_name)
            actions.append(current.action)
            load_nums.append(current.save_id)
        current = current.parent

    action_names.reverse()
    actions.reverse()
    load_nums.reverse()

    # if load_nums != None:
    #     print(f" LOAD: {load_nums[0]}, ACTION: {action_names[0]}")
    #     score, done, dead, damaged = simulate_action(save_num_list, env, actions_data["stay_still"]["action"], load_nums[0])
    #     keyboard.wait("space")

    print("Action path:")
    for i, a in enumerate(action_names):
        if show_steps:
            score, done, dead, damaged = simulate_action(save_num_list, env, actions[i], load_nums[i])
            dead_count += dead
            damaged_count += damaged
        print(f"  DEAD: {dead}: DAMAGED: {damaged}")
        print(f"  {i}: {str(load_nums[i])}: {a}: DEAD COUNT: {dead_count}: DAMAGED COUNT: {damaged_count}")
        if show_steps:
            keyboard.wait("space")

def simulate_action_path(actions_data, save_num_list, env, node, show_steps, save_load_num):
    action_names = []
    actions = []
    load_nums = []
    current = node

    final_score = 0
    dead_count = 0
    damaged_count = 0

    print("BASE LOAD: " + str(save_load_num))
    while current is not None:
        if current.action_name is not None:
            action_names.append(current.action_name)
            actions.append(current.action)
            load_nums.append(current.save_id)

            print("CURRENT LOAD: " + str(current.save_id))
        current = current.parent

    action_names.reverse()
    actions.reverse()
    load_nums.reverse()

    print("Action path:")
    # keyboard.wait("space")
    score, done, dead, damaged = simulate_action(save_num_list, env, actions_data["stay_still"]["action"], save_load_num)

    if dead != None:
        final_score += score
        dead_count += dead
        damaged_count += damaged
        print(f"  DEAD: {dead}: DAMAGED: {damaged}")
        print(f"  BASE: {str(save_load_num)}: stay_still: DEAD COUNT: {dead_count}: DAMAGED COUNT: {damaged_count}")
        if show_steps:
            keyboard.wait("space")

    for i, a in enumerate(action_names):
        score, done, dead, damaged = simulate_action(save_num_list, env, actions[i], load_nums[i])

        if dead != None:
            final_score += score
            dead_count += dead
            damaged_count += damaged
            print(f"  DEAD: {dead}: DAMAGED: {damaged}")
            print(f"  {i}: {str(load_nums[i])}: {a}: DEAD COUNT: {dead_count}: DAMAGED COUNT: {damaged_count}")
            if show_steps:
                keyboard.wait("space")

    return final_score, done, dead_count, damaged_count


def astar_search(save_num_list, env, selected_save_id, actions_data, max_depth):
    open_set = []
    visited = set()

    env.step(actions_data["stay_still"]["action"], -selected_save_id)
    save_num_list.append(selected_save_id)

    # print("Load start astar_search")
    # keyboard.wait("space") 

    root = Node(parent=None,
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

        # print("open_set: " + str(open_set))
        node = heapq.heappop(open_set)
        
        if node.depth >= max_depth:
            # print("AT LIMIT node.depth: " + str(node.depth))
            continue
        # print("GOOD node.depth: " + str(node.depth))

        # print("save_id: " + str(save_id))
        

        for name, entry in actions_data.items():
            selected_save_id += 1
            # env.step(actions_data["stay_still"]["action"], save_id)
            score, done, dead, damaged = simulate_action(save_num_list, env, entry["action"], node.save_id)

            

            child = Node(parent=node,
                        action_name=name,
                        action=entry["action"],
                        depth=node.depth + 1,
                        score=node.score + score,
                        dead=node.dead + dead,
                        damaged=node.damaged + damaged,
                        save_id=selected_save_id
                        )
            
            # score, done, dead, damaged = simulate_action_path(actions_data, save_num_list, env, child, False, node.save_id)
            # child.score = node.score + score
            # child.dead = node.dead + dead
            # child.damaged = node.damaged + damaged
            
            # if child.dead > 0:
            #     print("THIS IS DEAD")
            #     print("child dead: " + str(child.dead))
            #     keyboard.wait("space")

            # print("OPTION")
            # print_action_path(child)
            # print("SCORE: " + str(child.score))
            # print("----------")
            
            # if child.dead > 0 or child.damaged > 0:
            #     print("!!!!!!!!!!!!!")
            #     print("DAMAGED OPTION")
            #     print_action_path(save_num_list, env, child, False)
            #     print("DAMAGED SCORE: " + str(child.score))
            #     print("DAMAGED DEAD: " + str(child.dead))
            #     print("DAMAGED DAMAGED: " + str(child.damaged))
            #     print("!!!!!!!!!!!!!")
                # keyboard.wait("space") 

            if child.score > best_node.score:
                print("=============")

                if child.dead > 0:
                    print("2 THIS IS DEAD")
                    print("child dead: " + str(child.dead))
                    # keyboard.wait("space")

                print("BEST OPTION")
                
                print("PREV SCORE: " + str(best_node.score))
                print("BEST SCORE: " + str(child.score))
                print("BEST DEAD: " + str(child.dead))
                print("BEST DAMAGED: " + str(child.damaged))

                # test_child = child
                # test_damage = 0
                # test_dead = 0
                # while test_child is not None:
                #     test_damage += test_child.damaged
                #     print("test_damage: " + str(test_damage))
                #     test_dead += test_child.dead
                #     print("test_death: " + str(test_dead))
                #     print(f"  {str(test_child.save_id)}: {test_child.action_name}: DEAD: {test_child.dead}: DAMAGED: {test_child.damaged}")
                #     test_child = test_child.parent

                # simulate_action_path(actions_data, save_num_list, env, child, True, node.save_id)

                raw_grid, state, reward, done, info = env.step(actions_data["stay_still"]["action"], node.save_id)
                print("LOAD POSITION: " + str(node.save_id))
                # keyboard.wait("space")
                score, done, dead, damaged = simulate_action(save_num_list, env, entry["action"], node.save_id)
                print(f"MOVE BEST MOVEMENT: {name}, SAVE: {selected_save_id}")
                # keyboard.wait("space")
                
                # print_action_path(actions_data, save_num_list, env, child, True)
                print("=============")
                best_node = child

            if not done:
                heapq.heappush(open_set, child)
            # keyboard.wait("space")

            env.step(actions_data["stay_still"]["action"], -selected_save_id)
            save_num_list.append(selected_save_id)
            # print("Save potential action")
            # keyboard.wait("space") 

    # print("CHOSEN ACTION")
    # keyboard.wait("space")
    # print_action_path(save_num_list, env, best_node, True)

    action_plan, save_plan = extract_plan(best_node)
    # print("best_node: " + str(best_node.save_id))
    # print("selected_save_id: " + str(selected_save_id))
    return best_node, action_plan, save_plan, selected_save_id, save_num_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a PPO model for Solid Rally Rally Single Objective RL.")
    parser.add_argument("--run", type=int, required=True, help="Run ID")
    parser.add_argument("--weight", type=float, required=True, help="Weight value for SORL reward")
    parser.add_argument("--game", required=True, help="Name of environment")
    parser.add_argument("--target_arousal", type=float, required=True, help="Target Arousal")
    parser.add_argument("--cluster", type=int, required=True, help="Cluster index for Arousal Persona")
    parser.add_argument("--periodic_ra", type=int, required=True,
                        help="Assign arousal rewards every 3 seconds, instead of synchronised with behavior.")
    parser.add_argument("--cv", required=True, type=int, help="0 for GameObs, 1 for CV")
    parser.add_argument("--headless", required=True, type=int, help="0 for headless mode, 1 for graphics mode")
    parser.add_argument("--logdir", required=True, help="Log directory for TensorBoard")
    parser.add_argument("--grayscale", required=True, type=int, help="0 for RGB, 1 for grayscale")
    parser.add_argument("--discretize", required=True, type=int,
                        help="0 for continuous, 1 for discretized observations")
    parser.add_argument("--algorithm", required=True, help="Algorithm to use for training")
    parser.add_argument("--policy", required=False, help="Policy to use for training for PPO agents")
    parser.add_argument("--use_gpu", required=True, help="Use device GPU for models", type=int)
    parser.add_argument("--classifier", required=True, help="Use classifier model and reward for training", type=int)
    parser.add_argument("--preference", required=False, help="Use preference model for training", type=int)
    parser.add_argument("--decision_period", required=False, help="Decision period for environments", type=int,
                        default=10)
    parser.add_argument("--max_retries", required=False, help="Max retries for Unity timeout recovery", type=int,
                        default=500)

    args = parser.parse_args()

    if args.use_gpu == 1:
        device = torch.device("cuda")
        print(device)
    else:
        device = torch.device("cpu")

    if args.algorithm.lower() == "ppo":
        if "lstm" in args.policy.lower():
            pass
        else:
            model_class = PPO
    elif args.algorithm.lower() == "dqn":
        args.policy = "DQN"
        if args.cv == 0:
            model_class = RainbowAgent
        else:
            print("Model not implemented yet! Aborting...")
            exit()

    for run in range(args.run):

        print(f"\n{'=' * 60}")
        print(f"Starting Run {run}")
        print(f"{'=' * 60}\n")

        model = None

        try:
            main_actions_data = {
                "stay_still": {"action": (1, 0, 0), "score": 0},
                "move_left": {"action": (0, 0, 0), "score": 0},
                "move_right": {"action": (2, 0, 0), "score": 0},
                "jump_straight": {"action": (1, 1, 0), "score": 0},
                "jump_left": {"action": (0, 1, 0), "score": 0},
                "jump_right": {"action": (2, 1, 0), "score": 0},
            }

            main_tiles = {
                "wall": 4,
                "enemy": 1,
                "air": 5,
                "player": 0,
                "pickup": 2,
                "pickup_block": 3
            }

            main_env = create_environment(args, run)
            experiment_name = f'{args.logdir}/{args.game}/{"Ordinal" if args.preference else "Raw"}/{"Classification" if args.classifier == 1 else "Regression"}/{"Maximize Arousal" if args.target_arousal == 1 else "Minimize Arousal"}/{args.algorithm}/{args.policy}-Cluster{args.cluster}-{args.weight}λ-run{run}'
            main_env.callback = TensorBoardCallback(experiment_name, main_env, model)
            obs = main_env.reset()
            main_save_load_num = 1
            main_save_num_list = []

            main_raw_grid, main_state, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], -main_save_load_num)
            main_save_num_list.append(main_save_load_num)

            temprows, tempcols, t = main_raw_grid.shape
            main_player_pos = ((int((temprows - 1) / 2)), (int((tempcols - 1) / 2)))

            testCount = 0            
            
            main_pos_x = 0
            # print("POS_X 1: " + str(pos_x))

            while True:   
                main_new_save_load_num = main_save_load_num
                # print("state[37]: " + str(state[37]))
 
                main_best_node, main_action_plan, main_save_plan, main_new_save_load_num, main_save_num_list = astar_search(main_save_num_list, main_env, main_save_load_num, main_actions_data, max_depth=3)

                print("Action Plan: " + str(main_action_plan))
                print("Save Plan: " + str(main_save_plan))
                # keyboard.wait("space")

                # print_action_path(main_actions_data, main_save_num_list, main_env, main_best_node, True)
                score, done, dead, damaged = simulate_action(main_save_num_list, main_env, main_best_node.action, main_best_node.parent.save_id)
                keyboard.wait("space")

                # raw_grid, state, reward, done, info = env.step(actions_data["stay_still"]["action"], save_load_num)
                # print("1 DEAD: " + str(state[37]))
                # keyboard.wait("space")
                # print("Plan: " + str(main_plan))
                # # keyboard.wait("space")
                # if main_plan:
                #     for i, main_action in enumerate(main_plan):
                        
                #         # keyboard.wait("space")
                #         if i == 0:
                #             main_raw_grid, main_state, main_reward, main_d, main_info = main_env.step(main_action, main_save_load_num)
                #             print("MAIN")
                #             check_save_num(main_env, main_save_load_num, main_save_num_list)

                #             # print("First Real Action")
                #             # keyboard.wait("space")
                #         else:
                #             main_raw_grid, main_state, main_reward, main_d, main_info = main_env.step(main_action, 0)

                #         # print("PREV POS_X: " + str(pos_x))
                #         # print("state[0]: " + str(state[0]))
                #         main_pos_x += main_state[0]
                #         # print("POS_X 2: " + str(pos_x))

                #         dead = main_state[37]
                #         damaged = main_state[19]
                #         print("Action: " + str(main_action))
                #         print("dead: " + str(dead))
                #         print("damaged: " + str(damaged))
                #         print("pos_x: " + str(main_pos_x))
                #         print("---------")
                #         keyboard.wait("space")
                #         if main_d:
                #             break

                main_save_load_num = main_new_save_load_num + 1
                # print("2 DEAD: " + str(state[37]))
                # print("Pos X: " + str(pos_x))
                # print("state[0]: " + str(state[0]))

                # if pos_x < 10:
                #     keyboard.wait("space")

                # if pos_x > 200:
                # keyboard.wait("space")                              

        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
            break

        except Exception as e:
            print(f"\n❌ Fatal error in run {run}: {e}")
            traceback.print_exc()

        finally:
            print("Cleaning up resources...")
            if main_env is not None:
                if hasattr(main_env, 'callback'):
                    close_callback_safely(main_env.callback)
                close_environment_safely(main_env)
            print(f"{'=' * 60}\n")



















# for name, entry in actions_data.items():
                #     print(name)
                #     entry["score"] = 0
                #     grid = np.array(np.squeeze(raw_grid))
                #     raw_grid, state, reward, d, info = env.step(actions_data["stay_still"]["action"], save_load_num) #LOAD

                #     print("Load Pos")
                #     print("state[37]: " + str(state[37]))
                #     print("stat[19]: " + str(state[19]))
                #     # keyboard.wait("space")
                #     raw_grid, state, reward, d, info = env.step(entry["action"], 0) # MOVE
                #     entry["score"] = (calculate_score(state, reward) + entry["score"]) 
                #     print("state[0]: " + str(state[0]))
                #     print("reward: " + str(reward))
                #     print("temp score: " + str(entry["score"]))
                #     print(" ")
                #     print("--------------")
                #     print("Play " + str(name))
                #     print("state[37]: " + str(state[37]))
                #     print("stat[19]: " + str(state[19]))
                #     # keyboard.wait("space")

                # best_name = max(actions_data, key=lambda k: actions_data[k]["score"])
                # best_action = actions_data[best_name]["action"]

                # print("============================================================")
                # print("Best Action: " + str(best_name))
                # print("Best Score: " + str(actions_data[best_name]["score"]))

                # print("POST 1: Load")
                # raw_grid, state, reward, d, info = env.step(actions_data["stay_still"]["action"], save_load_num) #LOAD
                
                # # keyboard.wait("space")

                # print("POST 2: Move")
                # raw_grid, state, reward, d, info = env.step(entry["action"], 0) #MOVE
                
                # keyboard.wait("space")

                # print("POST 3: Save")
                # save_load_num += 1
                # raw_grid, state, reward, d, info = env.step(best_action, -save_load_num) #SAVE

                # # keyboard.wait("space")
                
                # print("POST 4: Load")
                # raw_grid, state, reward, d, info = env.step(actions_data["stay_still"]["action"], save_load_num) #LOAD
                
                # # keyboard.wait("space")  

















# def extract_plan():
#     actions = []

#     if best_position == None:
#         for i in range(10):
#             actions.append(actions_data["move_right"]["action"])

#         return actions
    
#     current = best_position
#     while current.parent_node != None:
#         for i in range(current.repetitions):
#             actions.append(current.action)
#             actions.reverse()
#         if current.state[19] or current.state[37]:
#             require_replanning = True

#     return actions

# def start_search(env, original_save_load_num, repetitions):
#     start_pos = Node(None, repetitions, None)
#     start_pos.initializeRoot(env, original_save_load_num)

#     pos_pool = []
#     visited_states = []
#     pos_pool.append(start_pos.generate_children())

#     best_position = start_pos
#     furthest_position = start_pos

# def pick_best_pos():
#     best_pos = None
#     best_pos_cost = 10000000
#     for current in pos_pool:
#         current_cost = env.episode_length - current.pos_x # ??

#         if current_cost < best_pos_cost:
#             best_pos_cost = current_cost

#     pos_pool.remove(best_pos)

# def search():
#     current = best_position
#     current_good = False
#     max_right = 176

#     while len(pos_pool) != 0 and not current_good:
#         current = pick_best_pos(pos_pool)

#         if current == None:
#             return None

#         current_good = False
#         real_score = current.simluate_pos()

#         if real_score < 0:
#             continue
#         elif not current.is_in_visited_list and 

# def isInVisited():
#     time_diff = 5
#     x_diff = 2
#     y_diff = 2

#     for v in range(visited_states):

# def optimise():
#     plan_ahead = 2
#     steps_per_search = 2
    
#     raw_grid, state, reward, d, info = env.step(actions_data["stay_still"]["action"], -save_load_num) #SAVE
#     original_save_load_num = save_load_num
    
#     ticks_before_replanning = ticks_before_replanning - 1
#     require_replanning = False

#     if ticks_before_replanning <= 0 or len(current_action_plan) == 0 or require_replanning:
#         current_action_plan = extract_plan()
#         if len(current_action_plan) < plan_ahead:
#             plan_ahead = len(current_action_plan)

#         for i in range(plan_ahead):
#             raw_grid, state, reward, d, info = env.step(current_action_plan[i], 0) #MOVE

#         start_search(env, steps_per_search)
#         ticks_before_replanning = plan_ahead

#     if state[37] == 1:
#         raw_grid, state, reward, d, info = env.step(actions_data["stay_still"]["action"], original_save_load_num) #LOAD
#         start_search(env, original_save_load_num, steps_per_search)

#     search()

#     action = None
#     if len(current_action_plan) > 0:
#         action = current_action_plan.pop(0)

#     return action




            # best_position = -1
            # furthest_position = -1
            # ticks_before_replanning = 0
            # node_list = []
            # current_action_plan = []
            # root = Node(None, state, actions_data["stay_still"]["action"], 0)    

            # pos_pool = []
            # visited_states = []

            # optimise()
            # ASTAR ===================================================






























            # if testCount % 8 == 0:
                #     print("Load " + str(save_load_num))
                #     raw_grid, state, reward, d, info = env.step(actions_data["jump_right"]["action"], save_load_num)
                # elif testCount % 4 == 0:
                #     save_load_num += 1
                #     print("Save " + str(save_load_num))
                #     raw_grid, state, reward, d, info = env.step(actions_data["jump_right"]["action"], -save_load_num)
                # else:
                #     print("Move")
                #     raw_grid, state, reward, d, info = env.step(actions_data["jump_right"]["action"], 0)

                # testCount += 1
                # print("testCount: " + str(testCount))
                # keyboard.wait("space")














                 # for name, entry in actions_data.items():
                #     print(name)
                #     entry["score"] = 0
                #     grid = np.array(np.squeeze(raw_grid))
                #     raw_grid, state, reward, d, info = env.step(actions_data["stay_still"]["action"], save_load_num)

                #     # print("state[37]: " + str(state[37]))
                #     # print("state[38]: " + str(state[38]))
                #     # print("stat[39]: " + str(state[39]))
                #     # print("state[288]: " + str(state[288]))
                #     print("Load Pos")
                #     print("state[38]: " + str(state[38]))
                #     print("stat[39]: " + str(state[39]))
                #     keyboard.wait("space")
                #     raw_grid, state, reward, d, info = env.step(entry["action"], 0)
                #     entry["score"] = (calculate_score(state[0], reward) + entry["score"]) 
                #     print("state[0]: " + str(state[0]))
                #     print("reward: " + str(reward))
                #     print("temp score: " + str(entry["score"]))
                #     print(" ")
                #     print("--------------")
                #     print("Play " + str(name))
                #     print("state[38]: " + str(state[38]))
                #     print("stat[39]: " + str(state[39]))
                #     keyboard.wait("space")

                # best_name = max(actions_data, key=lambda k: actions_data[k]["score"])
                # best_action = actions_data[best_name]["action"]

                # print("============================================================")
                # print("Best Action: " + str(best_name))
                # print("Best Score: " + str(actions_data[best_name]["score"]))

                # raw_grid, state, reward, d, info = env.step(best_action, save_load_num)
                # print("Load Pos")
                # print("state[38]: " + str(state[38]))
                # print("stat[39]: " + str(state[39]))
                # keyboard.wait("space")
                # save_load_num += 1
                # raw_grid, state, reward, d, info = env.step(best_action, -save_load_num)
                # raw_grid, state, reward, d, info = env.step(best_action, 0)
                # # save_load_num += 1
                # # raw_grid, state, reward, d, info = env.step(best_action, -save_load_num)
                # # raw_grid, state, reward, d, info = env.step(best_action, 0)
                # # save_load_num += 1
                # # raw_grid, state, reward, d, info = env.step(best_action, -save_load_num)
                # # raw_grid, state, reward, d, info = env.step(best_action, 0)
                # # save_load_num += 1
                # # raw_grid, state, reward, d, info = env.step(best_action, -save_load_num)
                # # raw_grid, state, reward, d, info = env.step(best_action, 0)
                # # save_load_num += 1
                # # raw_grid, state, reward, d, info = env.step(best_action, -save_load_num)
                # # raw_grid, state, reward, d, info = env.step(best_action, 0)

                # print("Save Pos")
                # print("state[38]: " + str(state[38]))
                # print("stat[39]: " + str(state[39]))
                # keyboard.wait("space")
                # print("Load Pos")
                # print("state[38]: " + str(state[38]))
                # print("stat[39]: " + str(state[39]))
                # raw_grid, state, reward, d, info = env.step(actions_data["stay_still"]["action"], save_load_num)
                
                # keyboard.wait("space")




                


                # env.step(actions_data["stay_still"]["action"], save_load_num)
                # keyboard.wait("space")
                # grid = np.array(np.squeeze(raw_grid))
                # print("============================================================")
                # print("actions_data[jump_right][action]: " + str(actions_data["jump_right"]["action"]))
                # save_load_num += 1

                # if testCount % 8 == 0:
                #     print("Load " + str(save_load_num))
                #     raw_grid, state, reward, d, info = env.step(actions_data["jump_right"]["action"], save_load_num)
                # elif testCount % 4 == 0:
                #     save_load_num += 1
                #     print("Save " + str(save_load_num))
                #     raw_grid, state, reward, d, info = env.step(actions_data["jump_right"]["action"], -save_load_num)
                # else:
                #     print("Move")
                #     raw_grid, state, reward, d, info = env.step(actions_data["jump_right"]["action"], 0)
                
                
                # # print("state[0]: " + str(state[0]))
                # # print("reward: " + str(reward))
                # # print("final score: " + str(calculate_score(state[0], reward)))
                # # keyboard.wait("space")
                # # print("--------------")

                # testCount += 1
                # print("testCount: " + str(testCount))
                # keyboard.wait("space")































                # env.step(actions_data["stay_still"]["action"], save_load_num)
                
                # for name, entry in actions_data.items():
                #     grid = np.array(np.squeeze(raw_grid))
                #     # walkable, enemy_mask, point_mask, point_block_mask, pit_mask = create_walkable_and_enemy_mask(grid, player_pos, tiles)

                #     raw_grid, state, reward, d, info = env.step(entry["action"], 0)
                #     # raw_grid, state, reward, d, info = env.step(entry["action"], 0)
                    
                #     print("action name: " + str(name))
                #     print("state: " + str(state))
                #     print("state[0]: " + str(state[0]))
                #     print("reward: " + str(reward))
                #     entry["score"] = calculate_score(state[0], reward)
                #     print("final score: " + str(entry["score"]))
                #     keyboard.wait("space")
                #     print("--------------")

                #     env.step(actions_data["stay_still"]["action"], save_load_num)
                    

                # best_name = max(actions_data, key=lambda k: actions_data[k]["score"])
                # best_action = actions_data[best_name]["action"]
                # print("Best Action: " + str(best_name))
                # print("Best Score: " + str(actions_data[best_name]["score"]))
                # print("=================")
                # env.step(best_action, 0)
                # save_load_num += 1
                # env.step(actions_data["stay_still"]["action"], -save_load_num)
                # # keyboard.wait("space")
                # env.step(best_action, 0)
                # save_load_num += 1
                # env.step(actions_data["stay_still"]["action"], -save_load_num)
                # # keyboard.wait("space")
                # env.step(actions_data["stay_still"]["action"], save_load_num)
                # keyboard.wait("space")


                    # test = np.squeeze(raw_grid)
                    # test = np.array(test)
                    # # test[player_pos] = 0
                    # test = np.flipud(test)
                    # print("Grid:")
                    # print(str(test))

                    # if count < 5:
                    #     print("Continue")
                    #     count += 1
                    #     env.step(actions, 0)
                    # else:
                    #     print("Load")
                    #     env.step(actions[5], save_load_num)
                    #     count = 0