from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from mlagents_envs.exception import UnityTimeOutException

import argparse
import traceback

from affectively.environments.pirates_cv import PiratesEnvironmentCV
from affectively.environments.heist_cv import HeistEnvironmentCV
from affectively.environments.solid_cv import SolidEnvironmentCV
from affectively.environments.heist_game_obs import HeistEnvironmentGameObs
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.utils.logging import TensorBoardCallback
from agents.game_obs.Rainbow_DQN import RainbowAgent
import torch


def create_environment(args, run):
    """Factory function to create a new environment with the same parameters"""
    if args.cv == 0:
        if args.game == "fps":
            return HeistEnvironmentGameObs(
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
            return SolidEnvironmentGameObs(
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


def train_with_recovery(model, env, callbacks, total_timesteps, max_retries=5):
    """Train with automatic recovery from Unity timeout errors"""
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
            return True
            
        except UnityTimeOutException as e:
            retry_count += 1
            print(f"\n⚠️ Unity timeout at timestep {model.num_timesteps} (attempt {retry_count}/{max_retries})")
            print(f"Error: {e}")
            
            if retry_count < max_retries:
                print("🔄 Attempting to recover...")
                
                # Close the broken environment
                try:
                    env.env.close()
                except Exception as close_error:
                    print(f"Warning during env close: {close_error}")
                
                # Recreate the environment using the factory function
                # Note: You'll need to pass args and run to this function
                print("🔨 Creating new environment...")
                # This will be handled in the main loop where we have access to args
                return False  # Signal that we need to recreate
            else:
                print(f"❌ Max retries ({max_retries}) reached at timestep {model.num_timesteps}")
                raise
                
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            traceback.print_exc()
            raise


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
    parser.add_argument("--max_retries", required=False, help="Max retries for Unity timeout recovery", type=int, default=5)
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
        args.policy="DQN"
        if args.cv == 0:
            model_class = RainbowAgent
        else:
            print("Model not implemented yet! Aborting...")
            exit()

    for run in range(args.run):
        print(f"\n{'='*60}")
        print(f"Starting Run {run}")
        print(f"{'='*60}\n")
        
        # Create initial environment
        env = create_environment(args, run)
        
        # Create model
        model = model_class(policy=args.policy, env=env, device=device)
        
        # Setup experiment tracking
        experiment_name = f'{args.logdir}/{args.game}/{"Ordinal" if args.preference else "Raw"}/{"Classification" if args.classifier==1 else "Regression"}/{"Maximize Arousal" if args.target_arousal == 1 else "Minimize Arousal"}/{args.algorithm}/{args.policy}-Cluster{args.cluster}-{args.weight}λ-run{run}'
        env.callback = TensorBoardCallback(experiment_name, env, model)
        callbacks = ProgressBarCallback()
        
        # Train with automatic recovery
        training_complete = False
        recovery_attempts = 0
        max_recovery_attempts = args.max_retries
        
        while not training_complete and recovery_attempts < max_recovery_attempts:
            success = train_with_recovery(
                model=model,
                env=env,
                callbacks=callbacks,
                total_timesteps=5_000_000,
                max_retries=3  # Inner retries before recreating env
            )
            
            if success:
                training_complete = True
            else:
                # Need to recreate environment
                recovery_attempts += 1
                print(f"\n🔄 Recovery attempt {recovery_attempts}/{max_recovery_attempts}")
                
                try:
                    env.env.close()
                except:
                    pass
                
                # Create new environment
                env = create_environment(args, run)
                
                # Update model's environment
                model.set_env(env)
                env.callback = TensorBoardCallback(experiment_name, env, model)
                
                print(f"✅ Environment recreated, resuming from timestep {model.num_timesteps}")
        
        if training_complete:
            # Save the trained model
            model.save(f"{experiment_name}.zip")
            print(f"✅ Finished run {run} - Model saved!")
        else:
            print(f"❌ Run {run} failed after {recovery_attempts} recovery attempts")
        
        # Clean up
        try:
            env.env.close()
        except:
            pass