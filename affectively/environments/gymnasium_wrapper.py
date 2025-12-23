import numpy as np
import gym
import gymnasium as gymnasium

def to_gymnasium_space(space):
    # Discrete
    if isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(space.n)

    # Box
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=np.array(space.low),
            high=np.array(space.high),
            shape=space.shape,
            dtype=space.dtype,
        )

    # MultiDiscrete
    if isinstance(space, gym.spaces.MultiDiscrete):
        return gymnasium.spaces.MultiDiscrete(np.array(space.nvec))

    # MultiBinary
    if isinstance(space, gym.spaces.MultiBinary):
        return gymnasium.spaces.MultiBinary(space.n)

    # Dict / Tuple (if you ever use them)
    if isinstance(space, gym.spaces.Dict):
        return gymnasium.spaces.Dict({k: to_gymnasium_space(v) for k, v in space.spaces.items()})

    if isinstance(space, gym.spaces.Tuple):
        return gymnasium.spaces.Tuple(tuple(to_gymnasium_space(s) for s in space.spaces))

    raise TypeError(f"Unsupported space type: {type(space)}")


class GymToGymnasiumWrapper(gymnasium.Env):
    def __init__(self, env: gym.Env):
        self.env = env
        self.action_space = to_gymnasium_space(env.action_space)
        self.observation_space = to_gymnasium_space(env.observation_space)
        if hasattr(env, "reward_space"):
            self.reward_space = to_gymnasium_space(env.reward_space)

    def reset(self, *, seed=None, options=None):
        print(self.callback, self.env.episode_arousal_trace)
        if self.callback is not None and len(self.env.episode_arousal_trace) > 0:
            self.callback.on_episode_end()
        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        terminated = bool(done or getattr(self.env.unwrapped.customSideChannel, "levelEnd", False))
        truncated = bool(getattr(self.env.unwrapped, "episode_length", 0) > (6000 / getattr(self.env.unwrapped, "decision_period", 1)))

        if truncated:
            info["TimeLimit.truncated"] = True

        if getattr(self.env.unwrapped.customSideChannel, "levelEnd", False):
            self.env.unwrapped.customSideChannel.levelEnd = False

        # print(reward)
        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()
