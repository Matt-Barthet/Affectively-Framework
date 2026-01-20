import numpy as np
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from mlagents_envs.exception import UnityTimeOutException
from morl_baselines.multi_policy.envelope.envelope import Envelope

from tqdm import tqdm

import argparse
import traceback
import time

from affectively.environments.gymnasium_wrapper import GymToGymnasiumWrapper
from affectively.environments.pirates_cv import PiratesEnvironmentCV
from affectively.environments.heist_cv import HeistEnvironmentCV
from affectively.environments.solid_cv import SolidEnvironmentCV
from affectively.environments.heist_game_obs import HeistEnvironmentGameObs
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.utils.action_wrapper import FlattenMultiDiscreteAction
from affectively.utils.logging import TensorBoardCallback, MORLTensorBoardCallback
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
    parser.add_argument("--timesteps", required=False, help="Total timesteps for training", type=int, default=5_000_000)
    args = parser.parse_args()

    if args.use_gpu == 1:
        device = torch.device("cuda")
        print(device)
    else:
        device = torch.device("cpu")

    if args.algorithm.lower() == "ppo":
        model_class = PPO

    elif args.algorithm.lower() == "dqn":
        args.policy = "DQN"
        model_class = RainbowAgent
        if args.cv == 0:
            model_class = RainbowAgent
        else:
            print("Model not implemented yet! Aborting...")
            exit()

    elif args.algorithm.lower() in ["eq", "envelopeq", "envelope_q"]:
        model_class = "ENVELOPE_Q"
        args.weight = -1.0

    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    for run in range(args.run):

        print(f"\n{'=' * 60}")
        print(f"Starting Run {run}")
        print(f"{'=' * 60}\n")

        env = None
        model = None

        try:

            env = create_environment(args, run)
            env = GymToGymnasiumWrapper(env)
            experiment_name = f'{args.logdir}/{args.game}/{"Ordinal" if args.preference else "Raw"}/{"Classification" if args.classifier == 1 else "Regression"}/{"Maximize Arousal" if args.target_arousal == 1 else "Minimize Arousal"}/{args.algorithm}/{args.policy}-Cluster{args.cluster}-{args.weight}λ-run{run}'

            if model_class == "ENVELOPE_Q":

                eval_env = create_environment(args, run + 10001)
                env = FlattenMultiDiscreteAction(env)
                env = GymToGymnasiumWrapper(env)

                from gymnasium.envs.registration import EnvSpec

                env.spec = EnvSpec(
                    id=f"{args.game}-Unity-v0"
                )

                eval_env = GymToGymnasiumWrapper(FlattenMultiDiscreteAction(eval_env))
                agent = Envelope(
                    env=env,
                    log=True   # 👈 ADD THIS
                )

                env.callback = MORLTensorBoardCallback(
                    experiment_name,
                    env,
                    agent,
                    reference_point=np.asarray([-0.1, -0.1]),
                )

                print("📊 Starting Envelope-Q training")

                try:
                    agent.train(total_timesteps=args.timesteps, ref_point=np.asarray([-0.1, -0.1]), verbose=True, eval_env=eval_env, eval_freq=1_000_000, num_eval_episodes_for_front=2, num_eval_weights_for_eval=50, num_eval_weights_for_front=50)
                except UnityTimeOutException:
                    close_environment_safely(env)
                    raise

                agent.save(f"{experiment_name}.zip")
                print(f"✅ Finished run {run} - MORL agent saved!")


            else:
                model = model_class(policy=args.policy, env=env, device=device)
                env.callback = TensorBoardCallback(experiment_name, env, model)

                training_complete = False
                recovery_attempts = 0
                max_recovery_attempts = args.max_retries

                callbacks = PersistentProgressBarCallback(
                    total_timesteps=args.timesteps,
                    env_wrapper=env
                )

                while not training_complete and recovery_attempts < max_recovery_attempts:
                    success = train_with_recovery(
                        model=model,
                        env=env,
                        callbacks=callbacks,
                        total_timesteps=args.timesteps,
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
                        env = create_environment(args, run)
                        env = GymToGymnasiumWrapper(env)

                        if hasattr(model, 'set_env'):
                            model.set_env(env)
                            print("✓ Model environment updated")

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