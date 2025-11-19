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
    parser.add_argument("--decision_period", required=False, help="Decision period for environments", type=int, default=10)
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
                    decision_period=args.decision_period,
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
                    decision_period=args.decision_period,
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
                    decision_period=args.decision_period,
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
                    decision_period=args.decision_period,
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
                    decision_period=args.decision_period,
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
                    decision_period=args.decision_period,
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