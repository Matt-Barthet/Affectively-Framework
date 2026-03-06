import numpy as np
import gym
import gymnasium as gymnasium
from gym.spaces import MultiDiscrete, Discrete

from affectively.environments.heist_cv import HeistEnvironmentCV
from affectively.environments.heist_game_obs import HeistEnvironmentGameObs
from affectively.environments.pirates_cv import PiratesEnvironmentCV
from affectively.environments.pirates_game_obs import PiratesEnvironmentGameObs
from affectively.environments.solid_cv import SolidEnvironmentCV
from affectively.environments.solid_game_obs import SolidEnvironmentGameObs
from affectively.utils.logging import TensorboardGoExplore, TensorBoardCallback


def to_gymnasium_space(space):
    if isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(space.n)
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=np.array(space.low),
            high=np.array(space.high),
            shape=space.shape,
            dtype=space.dtype,
        )
    if isinstance(space, gym.spaces.MultiDiscrete):
        return gymnasium.spaces.MultiDiscrete(np.array(space.nvec))
    if isinstance(space, gym.spaces.MultiBinary):
        return gymnasium.spaces.MultiBinary(space.n)
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
        if self.env.callback is not None and len(self.env.episode_arousal_trace) > 0:
            self.env.callback.on_episode_end()
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

        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()


class FlattenMultiDiscreteAction(gym.ActionWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.action_space, MultiDiscrete):
            raise TypeError(f"FlattenMultiDiscreteAction requires MultiDiscrete, got {type(env.action_space)}")
        self.nvec = np.asarray(env.action_space.nvec, dtype=np.int64)
        self.action_space = Discrete(int(np.prod(self.nvec)))
        print(self.action_space)

        # Precompute mixed-radix multipliers for fast encode/decode
        self._radix = np.ones_like(self.nvec, dtype=np.int64)
        for i in range(len(self.nvec) - 2, -1, -1):
            self._radix[i] = self._radix[i + 1] * self.nvec[i + 1]

    def action(self, act: int):
        # scalar -> vector
        act = int(act)
        vec = (act // self._radix) % self.nvec
        return vec.astype(np.int64).tolist()

    def reverse_action(self, act):
        # vector -> scalar
        vec = np.asarray(act, dtype=np.int64)
        return int(np.dot(vec, self._radix))


def create_environment(args, run, callback=None):
    if args.cv == 0:
        if args.game == "fps":
            env = HeistEnvironmentGameObs(
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
                imitate=args.imitate
            )
        elif args.game == "solid":
            env = SolidEnvironmentGameObs(
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
                imitate=args.imitate

            )
        elif args.game == "platform":
            env = PiratesEnvironmentGameObs(
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
                imitate=args.imitate

            )
    elif args.cv == 1:
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
                imitate=args.imitate

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
                imitate=args.imitate

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
                imitate=args.imitate

            )
    return env


def close_environment_safely(env):
    try:
        if hasattr(env, 'env'):
            env.env.close()
        else:
            env.close()
        print("Environment closed")
    except Exception as e:
        print(f"Warning during env close: {e}")
    print("Waiting for ports to be released...")
