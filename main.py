import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from affectively.environments.pirates_cv import PiratesEnvironmentCV
from affectively.environments.heist_cv import HeistEnvironmentCV

if __name__ == "__main__":
    run = 2
    weight = 0

    env = HeistEnvironmentCV(
        id_number=run,
        weight=weight,
        graphics=True,
        logging=True,
        log_prefix="LSTM/",
        grayscale=False
    )

    env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # env = VecTransposeImage(env)  # Fix channel order

    policy_kwargs = dict(
        # features_extractor_class=CustomResNetExtractor,
        # features_extractor_kwargs=dict(features_dim=256),
        net_arch = dict(pi=[256, 256], vf=[256, 256]),
        activation_fn = torch.nn.ReLU,
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=True,
        enable_critic_lstm=False,
        normalize_images=False,
    )

    label = 'optimize' if weight == 0 else 'arousal' if weight == 1 else 'blended'
    callbacks = ProgressBarCallback()

    model = RecurrentPPO(
        policy="CnnLstmPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        tensorboard_log=".results/tensorboard/LSTM/",
        device='cuda',
        n_steps=600,
        clip_range=0.1,
        learning_rate=1e-4,
    )
    model.learn(total_timesteps=100_000_000, callback=callbacks)
    model.save(f"./agents/PPO/cnn_ppo_solid_{label}_{run}_extended")