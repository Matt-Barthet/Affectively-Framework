import argparse
import os
import traceback

import numpy as np
import torch
from mlagents_envs.exception import UnityTimeOutException
from morl_baselines.multi_policy.envelope.envelope import Envelope
from stable_baselines3.ppo import PPO

from affectively.environments import GymToGymnasiumWrapper, FlattenMultiDiscreteAction, close_environment_safely, \
    create_environment
from affectively.utils import init_parser
from affectively.utils.logging import MORLTensorBoardCallback, TensorboardGoExplore, \
    close_progress_bar_safely, PersistentProgressBarCallback, close_callback_safely, TensorBoardCallback
from agents import init_model
from agents.game_obs.go_explore.agent import Explorer


def train_with_recovery(model, callbacks, total_timesteps):
    try:
        print(f"Starting/resuming training at timestep: {model.num_timesteps}/{total_timesteps}")
        model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=False)
        print(f"Training completed successfully! Final timesteps: {model.num_timesteps}")
        close_progress_bar_safely(callbacks)
        return True
    except Exception as e:
        print(f"\nUnity timeout at timestep {model.num_timesteps}")
        print(f"\nError: {e}")
        traceback.print_exc()
        close_progress_bar_safely(callbacks)
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Experiment trainer arguments.")
    parser = init_parser(parser)
    args = parser.parse_args()
    agent_class = init_model(args)
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")

    experiment_folder = f'{args.logdir}/{args.game}/{f"Synchronized Reward" if not args.periodic_ra else "Asynchronized Reward"}/{"Ordinal" if args.preference else "Raw"}/{"Classification" if args.classifier == 1 else "Regression"}/{"Maximize Arousal" if args.target_arousal == 1 else "Minimize Arousal"}/{args.algorithm}/'
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)

    for run in range(args.run):

        print(f"\n{'=' * 60}\nStarting Run {run}\n{'=' * 60}\n")
        experiment_name = f'{experiment_folder}{args.policy}-Cluster{args.cluster}-{args.weight}λ-run{run}'
        if os.path.exists(f"{experiment_name}.zip"):
            print("Model exists, skipping...")
            continue
        if os.path.exists(f"{experiment_name}.lock"):
            print("Other experiment is running here, skipping...")
            continue
        os.open(f"{experiment_name}.lock", os.O_CREAT)


        env = create_environment(args, run)
        env = GymToGymnasiumWrapper(env)

        try:

            if agent_class == "ENVELOPE_Q":

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
                    log=True
                )

                env.callback = MORLTensorBoardCallback(
                    experiment_name,
                    env,
                    agent,
                    reference_point=np.asarray([-0.1, -0.1]),
                )

                print("Starting Envelope-Q training")

                try:
                    agent.train(total_timesteps=args.timesteps, ref_point=np.asarray([-0.1, -0.1]), verbose=True, eval_env=eval_env, eval_freq=1_000_000, num_eval_episodes_for_front=2, num_eval_weights_for_eval=50, num_eval_weights_for_front=50)
                except UnityTimeOutException:
                    close_environment_safely(env)
                    raise

                agent.save(f"{experiment_name}.zip")
                print(f"Finished run {run} - MORL agent saved!")

            else:

                model = agent_class(policy=args.policy, env=env, device=device)

                if agent_class == Explorer:
                    callback = TensorboardGoExplore(experiment_name, env, model)
                else:
                    callback = TensorBoardCallback(experiment_name, env, model)

                env.env.callback = callback

                if agent_class == Explorer:
                    print(experiment_name.split('/')[-1])
                    env.env.callback = TensorboardGoExplore(experiment_name, env, model)
                    model.logdir = experiment_name
                else:
                    for i in range(16000, 0, -1000):
                        if os.path.exists(f"{experiment_name}-Episode-{i}.zip"):
                            model.load(f"{experiment_name}-Episode-{i}.zip")
                            model.set_parameters(f"{experiment_name}-Episode-{i}.zip")
                            print(f"Loaded at timestep: {i}")
                            break

                training_complete = False
                recovery_attempts = 0
                max_recovery_attempts = args.max_retries

                callbacks = PersistentProgressBarCallback(
                    total_timesteps=args.timesteps,
                    env_wrapper=env
                )

                while not training_complete and recovery_attempts < max_recovery_attempts:

                    success = train_with_recovery(model=model, callbacks=callbacks, total_timesteps=args.timesteps)
                    if success:
                        training_complete = True
                    else:
                        recovery_attempts += 1
                        print(f"\nRecovery attempt {recovery_attempts}/{max_recovery_attempts}")
                        old_callback = env.callback if hasattr(env, 'callback') else None
                        close_callback_safely(old_callback)
                        close_environment_safely(env)

                        env = create_environment(args, run)
                        env = GymToGymnasiumWrapper(env)

                        if hasattr(model, 'set_env'):
                            model.set_env(env)

                        if old_callback is not None:
                            old_callback.env = env
                            env.env.callback = old_callback

                        callbacks.env_wrapper = env
                        print(f"Environment recreated, resuming from timestep {model.num_timesteps}")

                if training_complete:
                    model.save(f"{experiment_name}.zip")
                    print(f"Finished run {run} - Model saved!")
                else:
                    print(f"Run {run} failed after {recovery_attempts} recovery attempts")
                os.remove(f"{experiment_name}.lock")

        except Exception as e:
            print(f"\nFatal error in run {run}: {e}")
            traceback.print_exc()

        finally:
            print("Cleaning up resources...")
            if env is not None:
                if hasattr(env, 'callback'):
                    close_callback_safely(env.callback)
                close_environment_safely(env)
            print(f"{'=' * 60}\n")

