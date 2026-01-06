<<<<<<< Updated upstream
# fro÷ßm sb3_contrib import RecurrentPPO
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import argparse

from affectively.environments.pirates_cv import PiratesEnvironmentCV
from affectively.environments.heist_cv import HeistEnvironmentCV
from affectively.environments.solid_cv import SolidEnvironmentCV
from affectively.environments.heist_game_obs import HeistEnvironmentGameObs
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.utils.logging import TensorBoardCallback
from agents.game_obs.Rainbow_DQN import RainbowAgent
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a PPO model for Solid Rally Rally Single Objective RL.")
    parser.add_argument("--run", type=int, required=True, help="Run ID")
    parser.add_argument("--weight", type=float, required=True, help="Weight value for SORL reward")
    parser.add_argument("--game", required=True, help="Name of environment")
    parser.add_argument("--target_arousal", type=float, required=True, help="Target Arousal")
    parser.add_argument("--cluster", type=int, required=True, help="Cluster index for Arousal Persona")
    parser.add_argument("--periodic_ra", type=int, required=True, help="Assign arousal rewards every 3 seconds, instead of synchronised with behavior.")
    parser.add_argument("--cv", required=True, type=int, help="0 for GameObs, 1 for CV")
    parser.add_argument("--headless", required=True, type=int, help="0 for headless mode, 1 for graphics mode")
    parser.add_argument("--logdir", required=True, help="Log directory for TensorBoard")
    parser.add_argument("--grayscale", required=True, type=int, help="0 for RGB, 1 for grayscale")
    parser.add_argument("--discretize", required=True, type=int, help="0 for continuous, 1 for discretized observations")
    parser.add_argument("--algorithm", required=True, help="Algorithm to use for training")
    parser.add_argument("--policy", required=False, help="Policy to use for training for PPO agents")
    parser.add_argument("--use_gpu", required=True, help="Use device GPU for models", type=int)
    parser.add_argument("--classifier", required=True, help="Use classifier model and reward for training", type=int)
    parser.add_argument("--preference", required=False, help="Use preference model for training", type=int)
    args = parser.parse_args()

    if args.use_gpu == 1:
        device = torch.device("cuda")
        print(device)
    else:
        device = torch.device("cpu")

    if args.algorithm.lower() == "ppo":
        if "lstm" in args.policy.lower():
            # model_class = RecurrentPPO
            pass
        else:
            model_class = PPO
    elif args.algorithm.lower() == "dqn":
        args.policy="DQN"
        if args.cv == 0:
            model_class = RainbowAgent
        else:
            print("Model not implemented yet! Aborting...")
            exit()

    for run in range(args.run):
        if args.cv == 0:
            if args.game == "fps":
                env = HeistEnvironmentGameObs(
                    id_number=run,
                    weight=args.weight,
                    graphics=args.headless==0,
                    cluster=args.cluster,
                    target_arousal=args.target_arousal,
                    period_ra=args.periodic_ra,
                    discretize=args.discretize,
                    classifier=args.classifier,
                    preference=args.preference,
                )
            elif args.game == "solid":
                env = SolidEnvironmentGameObs(
                    id_number=run,
                    weight=args.weight,
                    graphics=args.headless==0,
                    cluster=args.cluster,
                    target_arousal=args.target_arousal,
                    period_ra=args.periodic_ra,
                    discretize=args.discretize,
                    classifier=args.classifier,
                    preference=args.preference,
                )
            elif args.game == "platform":
                env = PiratesEnvironmentGameObs(
                    id_number=run,
                    weight=args.weight,
                    graphics=True, # Pirates is bugged in headless, prevent it manually for now
                    cluster=args.cluster,
                    target_arousal=args.target_arousal,
                    period_ra=args.periodic_ra,
                    discretize=args.discretize,
                    classifier=args.classifier,
                    preference=args.preference,
                )
        elif args.cv == 1:  # CV builds cannot run in headless mode - the unity renderer must be switched on to produce frames.
            if args.game == "fps":
                env = HeistEnvironmentCV(
                    id_number=run,
                    weight=args.weight,
                    cluster=args.cluster,
                    target_arousal=args.target_arousal,
                    period_ra=args.periodic_ra,
                    grayscale=args.grayscale,
                    classifier=args.classifier,
                    preference=args.preference,
                )
            elif args.game == "solid":
                env = SolidEnvironmentCV(
                    id_number=run,
                    weight=args.weight,
                    cluster=args.cluster,
                    target_arousal=args.target_arousal,
                    period_ra=args.periodic_ra,
                    grayscale=args.grayscale,
                    classifier=args.classifier,
                    preference=args.preference,
                )
            elif args.game == "platform":
                env = PiratesEnvironmentCV(
                    id_number=run,
                    weight=args.weight,
                    cluster=args.cluster,
                    target_arousal=args.target_arousal,
                    period_ra=args.periodic_ra,
                    grayscale=args.grayscale,
                    classifier=args.classifier,
                    preference=args.preference,
                )

        model = model_class(policy=args.policy, env = env, device=device) # define model for training using pixels here
        experiment_name = f'{args.logdir}/{args.game}/{"Ordinal" if args.preference else "Raw"}/{"Classification" if args.classifier==1 else "Regression"}/{"Maximize Arousal" if args.target_arousal == 1 else "Minimize Arousal"}/{args.algorithm}/{args.policy}-Cluster{args.cluster}-{args.weight}λ-run{run}'
        env.callback =  TensorBoardCallback(experiment_name, env, model)
        label = 'optimize' if args.weight == 0 else 'arousal' if args.weight == 1 else 'blended'
        callbacks = ProgressBarCallback()

        model.learn(total_timesteps=5_000_000, callback=callbacks, reset_num_timesteps=False)
        model.save(f"{experiment_name}.zip")
        print(f"Finished run {run}")
        env.env.close()
=======
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
from helper import Helper
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

def extract_plan(best_pos, require_replanning):
    actions = []
    pos_x = []
    if best_pos == None:
        for i in range(10):
            actions.append(main_actions_data["move_right"]["action"])
            pos_x.append(1)
        return actions, pos_x, require_replanning
    
    current = best_pos
    while current.parent != None:
        for i in range(current.repetitions):
            actions.append(current.action)
            pos_x.append(current.pos_x)

        if current.state[19] > 0 or current.state[37] > 0:
            require_replanning = True
        
        current = current.parent

    actions.reverse()
    pos_x.reverse()
    return actions, pos_x, require_replanning

def start_search(env, dist_x, dist_y, damage, death, starting_state, starting_save_num, latest_save_num, starting_repetitions):
    print("start_search")
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
    
    pos_pool = []
    visited_states = []

    children, latest_save_num = start_pos.generate_children(env, starting_save_num, latest_save_num)

    for child in children:
        heapq.heappush(pos_pool, (child.calculate_cost(), child))
        
    current_starting_pos_x = start_pos.pos_x

    best_pos = start_pos
    furthest_pos = start_pos

    return best_pos, furthest_pos, current_starting_pos_x, pos_pool, visited_states, latest_save_num

def pick_best_pos(pos_pool):
    # print("pick_best_pos")
    best_pos_pool = None
    best_pos_cost = float("inf")

    for i, current_pos_pool in enumerate(pos_pool):
        current_cost = current_pos_pool[1].calculate_cost()
        if current_cost < best_pos_cost:
            best_pos_pool = current_pos_pool
            best_pos_cost = current_cost
            best_index = i
    
    print("Best Cost: " + str(best_pos_cost))
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
            # print("V X: " + str(v[0]))
            # print("POS X: " + str(x))
            # print("V Y: " + str(v[1]))
            # print("POS Y: " + str(y))
            # print("V T: " + str(v[2]))
            # print("TIME DIFF: " + str(t))
            # keyboard.wait("space") 
            return True
        
    return False

def search(env, pos_pool, best_pos, furthest_pos, current_starting_pos_x, latest_save_num, visited_states, require_replanning):
    print("search")
    current = best_pos
    current_good = False
    max_right = 20
    search_count = 0
    # print("1 search, extract_plan(best_pos, require_replanning): " + str(extract_plan(best_pos, require_replanning)))
    while search_count <= 600 and (len(pos_pool) != 0 and (((best_pos.pos_x - current_starting_pos_x) < max_right) or not current_good) and env.episode_length < 600):
        # print("START len(pos_pool): " + str(len(pos_pool)))
        # print("START best_pos.pos_x: " + str(best_pos.pos_x))
        # print("START current_starting_pos_x: " + str(current_starting_pos_x))
        # print("START current_good: " + str(current_good))
        
        current, pos_pool = pick_best_pos(pos_pool)
        
        if current == None:
            return None
        
        current_good = False
        real_remaining_time, latest_save_num = current.simulate_pos(env, latest_save_num, best_pos.remaining_time_estimated)
        # if best_pos.remaining_time_estimated > current.remaining_time_estimated:
        #     print("Better Simulated")
        #     keyboard.wait("space") 

        # print("real_remaining_time: " + str(real_remaining_time))
        # print("current.remaining_time_estimated: " + str(current.remaining_time_estimated))
        # print("was Visited: " + str(is_in_visited(current.pos_x, current.pos_y, current.time_elapsed, visited_states)))
        # keyboard.wait("space")
        # print("1 real_remaining_time: " + str(real_remaining_time))

        check_condition = -1

        if is_in_visited(current.pos_x, current.pos_y, current.time_elapsed, visited_states):
            current.penalty += Helper.visited_list_penalty
        
        if real_remaining_time < 0:
            # print("check 1")
            check_condition = 1
            continue
        elif current.damage > 0 or current.death > 0:
            check_condition = 2
            current.penalty += (Helper.visited_list_penalty * 3)
            heapq.heappush(pos_pool, (current.calculate_cost(), current))
        elif not current.is_in_visited_list and is_in_visited(current.pos_x, current.pos_y, current.time_elapsed, visited_states):
            check_condition = 3

            current_good = True

            # print("check 2")
            # current.penalty += Helper.visited_list_penalty
            # print("2 real_remaining_time: " + str(real_remaining_time))
            current.is_in_visited_list = True
            # current.remaining_time = real_remaining_time
            # current.remaining_time_estimated = real_remaining_time
            heapq.heappush(pos_pool, (current.calculate_cost(), current))
        # elif (real_remaining_time - current.remaining_time_estimated) > 0.1:
        #     print("check 3")
        #     current.remaining_time_estimated = real_remaining_time
        #     heapq.heappush(pos_pool, (current.calculate_cost(), current))
        else:
            check_condition = 4
            # print("check 4")
            current_good = True
            visited_states = visited(current.pos_x, current.pos_y, current.time_elapsed, visited_states)

            children, latest_save_num = current.generate_children(env, current.save_num, latest_save_num)

            for child in children:
                heapq.heappush(pos_pool, (child.calculate_cost(), child))

        if best_pos.pos_x < current.pos_x:
            print("****************************************")
            print("check_condition: " + str(check_condition)) 
            print("best_pos.pos_x: " + str(best_pos.pos_x))
            print("current.pos_x: " + str(current.pos_x))
            print("abs(best_pos.pos_x - current_starting_pos_x): " + str(abs(best_pos.pos_x - current_starting_pos_x)))
            print("abs(current.pos_x - current_starting_pos_x): " + str(abs(current.pos_x - current_starting_pos_x)))
            print("max_right: " + str(max_right))
            print("-----")
            print("current_good: " + str(current_good))
            print("current.damage: " + str(current.damage))
            print("current.death: " + str(current.death))
            print("-----")
            print("best_pos.remaining_time_estimated: " + str(best_pos.remaining_time_estimated))
            print("current.remaining_time_estimated: " + str(current.remaining_time_estimated))
            print("****************************************")

        if current_good:
            if current.damage == 0 and current.death == 0:
                if best_pos.remaining_time_estimated > current.remaining_time_estimated:
                    print("OLD extract_plan(best_pos, require_replanning): " + str(extract_plan(best_pos, require_replanning)))
                    print("OLD best_pos.remaining_time_estimated: " + str(best_pos.remaining_time_estimated))
                    best_pos = current
                    print("NEW extract_plan(best_pos, require_replanning): " + str(extract_plan(best_pos, require_replanning)))
                    print("NEW best_pos.remaining_time_estimated: " + str(best_pos.remaining_time_estimated))
                    print("=================================================================================")

                    # run_pos()
                    # keyboard.wait("space") 

            if current.pos_x > furthest_pos.pos_x:
                furthest_pos = current


        # print("Best Pos X: " + str(best_pos.pos_x) + ": Current Pos X: " + str(current_starting_pos_x))
        # print("current extract_plan(best_pos, require_replanning): " + str(extract_plan(best_pos, require_replanning)))
        # print("----")
        # print("check len(pos_pool): " + str(len(pos_pool)))
        # print("CHECK len(pos_pool) != 0: " + str(len(pos_pool) != 0))
        # print("----")
        # print("check best_pos.pos_x: " + str(best_pos.pos_x))
        # print("check current_starting_pos_x: " + str(current_starting_pos_x))
        # print("check abs(best_pos.pos_x - current_starting_pos_x): " + str(abs(best_pos.pos_x - current_starting_pos_x)))
        # print("check max_right: " + str(max_right))
        # print("CHECK abs(best_pos.pos_x - current_starting_pos_x) < max_right): " + str(abs(best_pos.pos_x - current_starting_pos_x) < max_right))
        # print("----")
        # print("check current_good: " + str(current_good))
        # print("CHECK not current_good: " + str(not current_good))
        # print("----")
        # print("check env.episode_length: " + str(env.episode_length))
        # print("CHECK env.episode_length < 600: " + str(env.episode_length < 600))
        # print("=======================================")
        # keyboard.wait("space") 
        search_count+=1
        print("search_count: " + str(search_count))
        

    if (current.pos_x - current_starting_pos_x) < max_right and furthest_pos.pos_x > best_pos.pos_x + 20:
        best_pos = furthest_pos

    print("FINAL search, extract_plan(best_pos, require_replanning): " + str(extract_plan(best_pos, require_replanning)))
    return best_pos, furthest_pos, pos_pool, visited_states, latest_save_num

def optimise(env, original_state, original_save_num, current_action_plan, original_dist_x, original_dist_y, original_damage, original_death):
    print("==================================================================================")
    print("==================================================================================")
    print("optimise")

    plan_ahead = 2
    steps_per_search = 1

    require_replanning = False

    latest_save_num = original_save_num

    # print("1 current_action_plan: " + str(current_action_plan)) 

    state = original_state

    # print("len(current_action_plan): " + str(len(current_action_plan))) 
    # print("require_replanning: " + str(require_replanning))
    # print("---")
    
    # print("2 current_action_plan: " + str(current_action_plan)) 

    if len(current_action_plan) < plan_ahead:
        plan_ahead = len(current_action_plan)

    best_pos, furthest_pos, current_starting_pos_x, pos_pool, visited_states, latest_save_num = start_search(env, original_dist_x, original_dist_y, original_damage, original_death, state, latest_save_num, latest_save_num, steps_per_search)
    # print("2 extract_plan(best_pos, require_replanning): " + str(extract_plan(best_pos, require_replanning)))

    if state[37] > 0:
        best_pos, furthest_pos, current_starting_pos_x, pos_pool, visited_states, latest_save_num = start_search(env, original_dist_x, original_dist_y, original_damage, original_death, original_state, original_save_num, latest_save_num, steps_per_search)
        # print("dead extract_plan(best_pos, require_replanning): " + str(extract_plan(best_pos, require_replanning)))

    best_pos, furthest_pos, pos_pool, visited_states, latest_save_num = search(env, pos_pool, best_pos, furthest_pos, current_starting_pos_x, latest_save_num, visited_states, require_replanning)
    
    current_action_plan, test_pos_x, require_replanning = extract_plan(best_pos, require_replanning)

    # print("3 current_action_plan: " + str(current_action_plan))
    # print("3 test_pos_x: " + str(test_pos_x))
    # print("3 extract_plan(best_pos, require_replanning): " + str(extract_plan(best_pos, require_replanning)))

    return current_action_plan, latest_save_num

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
                "stay_still": {"action": (1, 0, 0), "cost": 0},
                "move_left": {"action": (0, 0, 0), "cost": 0},
                "move_right": {"action": (2, 0, 0), "cost": 0},
                "jump_straight": {"action": (1, 1, 0), "cost": 0},
                "jump_left": {"action": (0, 1, 0), "cost": 0},
                "jump_right": {"action": (2, 1, 0), "cost": 0},
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
            main_save_load_num = 0

            main_save_load_num += 1
            main_env.step(main_actions_data["stay_still"]["action"], -main_save_load_num)
            print("intial save, main_save_load_num: " + str(main_save_load_num))
            keyboard.wait("space") 
            main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], main_save_load_num)
            print("intial load, main_save_load_num: " + str(main_save_load_num))
            keyboard.wait("space") 
            
            main_current_action_plan = []
            main_dist_x = 0
            main_dist_y = 0
            main_damage = 0
            main_death = 0

            while True:   
                main_new_save_load_num = main_save_load_num

                # action, best_pos, 
                # pos_pool, 
                # visited_states, ticks_before_replanning, 
                # latest_save_num

                main_current_action_plan = []
                main_current_action_plan, main_new_save_load_num = optimise(main_env, main_state, main_new_save_load_num, main_current_action_plan, main_dist_x, main_dist_y, main_damage, main_death)
                # main_env.step(main_actions_data["stay_still"]["action"], main_save_load_num)
                # print("Load to intial pos, main_save_load_num: " + str(main_save_load_num))
                # keyboard.wait("space") 

                if main_current_action_plan == None:
                    main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(main_actions_data["stay_still"]["action"], main_save_load_num)
                
                    main_dist_x += main_state[0]
                    main_dist_y += main_state[1]
                    main_damage += main_state[19]
                    main_death += main_state[37]
                else:
                    for i, part_action in enumerate(main_current_action_plan):
                        if i == 0:
                            main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(part_action, main_save_load_num)
                        elif i == (len(main_current_action_plan) - 1):
                            main_save_load_num = main_new_save_load_num + 1  
                            main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(part_action, -main_save_load_num)
                        else:
                            main_raw_grid, main_state, main_reached_termination, main_reached_end_door, main_reward, main_done, main_info = main_env.step(part_action, 0)
                        
                        print("PART ACTION: " + str(part_action))
                        print("OLD DIST X: " + str(main_dist_x))
                        print("MOVED: " + str(main_state[0]))
                        main_dist_x += main_state[0]
                        main_dist_y += main_state[1]
                        main_damage += main_state[19]
                        main_death += main_state[37]
                        print("PART DIST X: " + str(main_dist_x))
                        # print("PART DIST Y: " + str(main_dist_y))
                        # print("PART DAMAGE: " + str(main_damage))
                        # print("PART DEATH: " + str(main_death))

                        keyboard.wait("space")

                # print("Move")
                # keyboard.wait("space")
                              
                # main_save_load_num = main_new_save_load_num + 1  
                # main_env.step(main_actions_data["stay_still"]["action"], -main_save_load_num)

                print("Save")
                keyboard.wait("space")      

                test_raw_grid, test_state, test_reached_termination, test_reached_end_door, test_reward, test_done, test_info = main_env.step(main_actions_data["stay_still"]["action"], main_save_load_num)
                print("Load")
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
>>>>>>> Stashed changes
