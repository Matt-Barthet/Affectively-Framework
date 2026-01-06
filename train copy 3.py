from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from mlagents_envs.exception import UnityTimeOutException
from tqdm import tqdm

import argparse
import traceback
import time

# from affectively.environments.pirates_cv import PiratesEnvironmentCV
# from affectively.environments.heist_cv import HeistEnvironmentCV
# from affectively.environments.solid_cv import SolidEnvironmentCV
from affectively.environments.heist_game_obs import HeistEnvironmentGameObs
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.utils.logging import TensorBoardCallback
from agents.game_obs.Rainbow_DQN import RainbowAgent
import torch

import keyboard
import numpy as np

from node import AStarNode
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
    # elif args.cv == 1:
    #     if args.game == "fps":
    #         return HeistEnvironmentCV(
    #             id_number=run,
    #             weight=args.weight,
    #             cluster=args.cluster,
    #             target_arousal=args.target_arousal,
    #             period_ra=args.periodic_ra,
    #             grayscale=args.grayscale,
    #             classifier=args.classifier,
    #             preference=args.preference,
    #             decision_period=args.decision_period,
    #         )
    #     elif args.game == "solid":
    #         return SolidEnvironmentCV(
    #             id_number=run,
    #             weight=args.weight,
    #             cluster=args.cluster,
    #             target_arousal=args.target_arousal,
    #             period_ra=args.periodic_ra,
    #             grayscale=args.grayscale,
    #             classifier=args.classifier,
    #             preference=args.preference,
    #             decision_period=args.decision_period,
    #         )
    #     elif args.game == "platform":
    #         return PiratesEnvironmentCV(
    #             id_number=run,
    #             weight=args.weight,
    #             cluster=args.cluster,
    #             target_arousal=args.target_arousal,
    #             period_ra=args.periodic_ra,
    #             grayscale=args.grayscale,
    #             classifier=args.classifier,
    #             preference=args.preference,
    #             decision_period=args.decision_period,
    #         )
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

# def heuristic(state, reward):
#     dead = state[37]
#     damaged = state[19]

#     if dead:      # dead
#         return -9999, dead, damaged
#     forward = state[0]
#     return (forward + reward), dead, damaged

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
    raw_grid, state, reached_termination, reached_end_door, reward, done, info = env.step(action, save_load_num)
    print("1 delta: " + str(state[0]))
    # keyboard.wait("space")
    # print("SIMULATION")
    # check_save_num(env, save_load_num, save_num_list)

    # print("simulate action: " + str(action))
    # keyboard.wait("space")
    # score, dead, damaged = heuristic(state, reward)
        
    return done, dead, damaged, state


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

    # print("Action path:")
    for i, a in enumerate(action_names):
        if show_steps:
            done, dead, damaged, state = simulate_action(save_num_list, env, actions[i], load_nums[i])
            dead_count += dead
            damaged_count += damaged
        # print(f"  DEAD: {dead}: DAMAGED: {damaged}")
        # print(f"  {i}: {str(load_nums[i])}: {a}: DEAD COUNT: {dead_count}: DAMAGED COUNT: {damaged_count}")
        if show_steps:
            keyboard.wait("space")

# def simulate_action_path(actions_data, save_num_list, env, node, show_steps, save_load_num):
#     action_names = []
#     actions = []
#     load_nums = []
#     current = node

#     final_score = 0
#     dead_count = 0
#     damaged_count = 0

#     # print("BASE LOAD: " + str(save_load_num))
#     while current is not None:
#         if current.action_name is not None:
#             action_names.append(current.action_name)
#             actions.append(current.action)
#             load_nums.append(current.save_id)

#             # print("CURRENT LOAD: " + str(current.save_id))
#         current = current.parent

#     action_names.reverse()
#     actions.reverse()
#     load_nums.reverse()

#     # print("Action path:")
#     # keyboard.wait("space")
#     done, dead, damaged, state = simulate_action(save_num_list, env, actions_data["stay_still"]["action"], save_load_num)

#     if dead != None:
#         final_score += score
#         dead_count += dead
#         damaged_count += damaged
#         # print(f"  DEAD: {dead}: DAMAGED: {damaged}")
#         # print(f"  BASE: {str(save_load_num)}: stay_still: DEAD COUNT: {dead_count}: DAMAGED COUNT: {damaged_count}")
#         if show_steps:
#             keyboard.wait("space")

#     for i, a in enumerate(action_names):
#         done, dead, damaged, state = simulate_action(save_num_list, env, actions[i], load_nums[i])

#         if dead != None:
#             final_score += score
#             dead_count += dead
#             damaged_count += damaged
#             # print(f"  DEAD: {dead}: DAMAGED: {damaged}")
#             # print(f"  {i}: {str(load_nums[i])}: {a}: DEAD COUNT: {dead_count}: DAMAGED COUNT: {damaged_count}")
#             if show_steps:
#                 keyboard.wait("space")

#     return final_score, done, dead_count, damaged_count
    
def calculate_cost(elapsed_time, dist_to_end):
    cost = elapsed_time + dist_to_end
    return cost

def astar_search(save_num_list, env, selected_save_id, actions_data, max_depth):
    open_set = []

    raw_grid, state, reached_termination, reached_end_door, reward, done, info = env.step(actions_data["stay_still"]["action"], -selected_save_id)
    print("2 delta: " + str(state[0]))
    # keyboard.wait("space")
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
            done, dead, damaged, state = simulate_action(save_num_list, env, entry["action"], node.save_id)

            child = AStarNode(parent=node,
                        action_name=name,
                        action=entry["action"],
                        depth=node.depth + 1,
                        score=node.score + score,
                        dist_to_end= node.dist_to_end + state[0], 
                        elapsed_time=env.episode_length, 
                        cost=calculate_cost((env.episode_length / 600), (173 - (node.dist_to_end + state[0]))),
                        dead=node.dead + dead,
                        damaged=node.damaged + damaged,
                        save_id=selected_save_id
                        )

            if child.cost < best_node.cost:
                # print("=============")

                # if child.dead > 0:
                #     print("2 THIS IS DEAD")
                #     print("child dead: " + str(child.dead))

                # print("BEST OPTION")
                
                # print("PREV SCORE: " + str(best_node.score))
                # print("BEST SCORE: " + str(child.score))
                # print("BEST DEAD: " + str(child.dead))
                # print("BEST DAMAGED: " + str(child.damaged))

                raw_grid, state, reached_termination, reached_end_door, reward, done, info = env.step(actions_data["stay_still"]["action"], node.save_id)
                print("3 delta: " + str(state[0]))

                # keyboard.wait("space")
                # print("LOAD POSITION: " + str(node.save_id))
                done, dead, damaged, state = simulate_action(save_num_list, env, entry["action"], node.save_id)
                # print(f"MOVE BEST MOVEMENT: {name}, SAVE: {selected_save_id}")

                # print("=============")
                best_node = child

            if not done:
                heapq.heappush(open_set, child)

            raw_grid, state, reached_termination, reached_end_door, reward, done, info = env.step(actions_data["stay_still"]["action"], -selected_save_id)
            print("4 delta: " + str(state[0]))
            
            # keyboard.wait("space")
            save_num_list.append(selected_save_id)

    action_plan, save_plan = extract_plan(best_node)

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
            testing_counting = 0
            moving_counting = 0
            main_save_num_list = []

            testing_counting += 1
            moving_counting += 1
            # print(f"{moving_counting}: {testing_counting}: SAVE")
            main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], -testing_counting)
            print("5 delta: " + str(main_state[0]))
            
            # keyboard.wait("space")
            main_save_num_list.append(main_save_load_num)

            temprows, tempcols, t = main_raw_grid.shape
            main_player_pos = ((int((temprows - 1) / 2)), (int((tempcols - 1) / 2)))

            testCount = 0            
            
            main_pos_x = 0

            dist = 0
            while True:   
                # moving_counting += 1

                # if moving_counting % 2 == 0:
                #     print(f"{moving_counting}: {testing_counting}: LOAD")
                #     main_env.step(main_actions_data["jump_right"]["action"], testing_counting)
                #     keyboard.wait("space")
                # else:
                #     testing_counting += 1
                #     print(f"{moving_counting}: {testing_counting}: SAVE")
                #     main_env.step(main_actions_data["jump_right"]["action"], -testing_counting)
                #     keyboard.wait("space")

                # keyboard.wait("space")

                main_new_save_load_num = main_save_load_num
                main_best_node, main_action_plan, main_save_plan, main_new_save_load_num, main_save_num_list = astar_search(main_save_num_list, main_env, main_save_load_num, main_actions_data, max_depth=3)

                if main_action_plan != None and main_save_plan != None:
                    done, dead, damaged, state = simulate_action(main_save_num_list, main_env, main_best_node.action, main_best_node.parent.save_id)
                else:
                    done, dead, damaged, state = simulate_action(main_save_num_list, main_env, main_actions_data["stay_still"]["action"], main_save_load_num)

                
                dist = dist + state[0]
                print("dist: " + str(dist))
                main_save_load_num = main_new_save_load_num + 1  
                keyboard.wait("space")                           

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