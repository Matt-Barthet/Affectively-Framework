from celery import Celery

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


def create_environment(cv, game, run, weight, headless, cluster, target_arousal, periodic_ra, grayscale, classifier, preference, decision_period):
    if cv == 0:
        if game == "fps":
            return HeistEnvironmentGameObs(
                id_number=run,
                weight=weight,
                graphics=headless == 0,
                cluster=cluster,
                target_arousal=target_arousal,
                period_ra=periodic_ra,
                discretize=discretize,
                classifier=classifier,
                preference=preference,
                decision_period=decision_period,
            )
        elif game == "solid":
            return SolidEnvironmentGameObs(
                id_number=run,
                weight=weight,
                graphics=headless == 0,
                cluster=cluster,
                target_arousal=target_arousal,
                period_ra=periodic_ra,
                discretize=discretize,
                classifier=classifier,
                preference=preference,
                decision_period=decision_period,
            )
        elif game == "platform":
            return PiratesEnvironmentGameObs(
                id_number=run,
                weight=weight,
                graphics=True,
                cluster=cluster,
                target_arousal=target_arousal,
                period_ra=periodic_ra,
                discretize=discretize,
                classifier=classifier,
                preference=preference,
                decision_period=decision_period,
            )
    elif cv == 1:
        if game == "fps":
            return HeistEnvironmentCV(
                id_number=run,
                weight=weight,
                cluster=cluster,
                target_arousal=target_arousal,
                period_ra=periodic_ra,
                grayscale=grayscale,
                classifier=classifier,
                preference=preference,
                decision_period=decision_period,
            )
        elif game == "solid":
            return SolidEnvironmentCV(
                id_number=run,
                weight=weight,
                cluster=cluster,
                target_arousal=target_arousal,
                period_ra=periodic_ra,
                grayscale=grayscale,
                classifier=classifier,
                preference=preference,
                decision_period=decision_period,
            )
        elif game == "platform":
            return PiratesEnvironmentCV(
                id_number=run,
                weight=weight,
                cluster=cluster,
                target_arousal=target_arousal,
                period_ra=periodic_ra,
                grayscale=grayscale,
                classifier=classifier,
                preference=preference,
                decision_period=decision_period,
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

app = Celery('Celery_Train', broker='pyamqp://guest@localhost//')

@app.task
def train(run, weight, game, target_arousal, cluster, periodic_ra, cv, headless, logdir, grayscale,
          discretize, algorithm, policy, use_gpu, classifier, preference, decision_period=10, max_retries=500):


    if use_gpu == 1:
        device = torch.device("cuda")
        print(device)
    else:
        device = torch.device("cpu")

    if algorithm.lower() == "ppo":
        if "lstm" in policy.lower():
            pass
        else:
            model_class = PPO
    elif algorithm.lower() == "dqn":
        policy = "DQN"
        if cv == 0:
            model_class = RainbowAgent
        else:
            print("Model not implemented yet! Aborting...")
            exit()

    for run in range(run):

        print(f"\n{'=' * 60}")
        print(f"Starting Run {run}")
        print(f"{'=' * 60}\n")

        env = None
        model = None

        try:
            # Create initial environment
            env = create_environment(cv, game, run, weight, headless, cluster, target_arousal, periodic_ra, grayscale, classifier, preference, decision_period)

            # Create model
            model = model_class(policy=policy, env=env, device=device)

            # Setup experiment tracking
            experiment_name = f'{logdir}/{game}/{"Ordinal" if preference else "Raw"}/{"Classification" if classifier == 1 else "Regression"}/{"Maximize Arousal" if target_arousal == 1 else "Minimize Arousal"}/{algorithm}/{policy}-Cluster{cluster}-{weight}λ-run{run}'
            env.callback = TensorBoardCallback(experiment_name, env, model)

            # Train with automatic recovery
            training_complete = False
            recovery_attempts = 0
            max_recovery_attempts = max_retries

            # Create a single persistent progress bar callback with environment reference
            callbacks = PersistentProgressBarCallback(total_timesteps=5_000_000, env_wrapper=env)

            while not training_complete and recovery_attempts < max_recovery_attempts:
                success = train_with_recovery(
                    model=model,
                    env=env,
                    callbacks=callbacks,
                    total_timesteps=5_000_000,
                    max_retries=500
                )

                if success:
                    training_complete = True
                else:
                    recovery_attempts += 1

                    print(f"\n🔄 Recovery attempt {recovery_attempts}/{max_recovery_attempts}")
                    old_callback = env.callback if hasattr(env, 'callback') else None
                    close_callback_safely(old_callback)
                    close_environment_safely(env)

                    print("🔨 Creating new environment...")
                    env = create_environment(cv, game, run, weight, headless, cluster, target_arousal, periodic_ra, grayscale, classifier, preference, decision_period)

                    if hasattr(model, 'set_env'):
                        model.set_env(env)
                        print("✓ Model environment updated")
                    else:
                        print("⚠️ Model doesn't have set_env method, continuing anyway")

                    if old_callback is not None:
                        old_callback.env = env
                        env.callback = old_callback
                        print("✓ Reattached existing callback to new environment")

                    callbacks.env_wrapper = env
                    print(f"✅ Environment recreated, resuming from timestep {model.num_timesteps}")

            if training_complete:
                model.save(f"{experiment_name}.zip")
                print(f"✅ Finished run {run} - Model saved!")
            else:
                print(f"❌ Run {run} failed after {recovery_attempts} recovery attempts")

        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
            break

        except Exception as e:
            print(f"\n❌ Fatal error in run {run}: {e}")
            traceback.print_exc()

        finally:
            print("Cleaning up resources...")
            if env is not None:
                if hasattr(env, 'callback'):
                    close_callback_safely(env.callback)
                close_environment_safely(env)
            print(f"{'=' * 60}\n")


# configs/run_A.yaml

# General experiment settings
runs= 5
output_dir= "./results/"
game= "fps"
algorithm= "PPO"
policy= "MlpPolicy"
conda_env= "affect-envs"

# Hardware settings
use_gpu= 0
headless= 1

# Specific game and agent parameters for this run
weight= 0.5
cluster= 0
target_arousal= 1
classifier= 1
preference= 1

# Specific algorithm parameters
periodic_ra= 0
cv= 0
grayscale= 0
discretize= 0
decisionPeriod= 10

# for run in range((runs)):
#     train.delay(run, weight, game, target_arousal, cluster, periodic_ra, cv, headless, output_dir, grayscale,
#                 discretize, algorithm, policy, use_gpu, classifier, preference, decisionPeriod)
