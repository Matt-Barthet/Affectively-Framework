import gymnasium as gym
import numpy as np
from gym.spaces import MultiDiscrete, Discrete

#  === add this wrapper class somewhere above main ===
class FlattenMultiDiscreteAction(gym.ActionWrapper):
    """
    Converts MultiDiscrete([n0, n1, ... nk]) actions <-> Discrete(prod(nvec)).
    Works with algorithms that require a scalar discrete action (Envelope-Q, PCN, DQN-style, etc.).
    """
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