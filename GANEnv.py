import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from GANWrapper import GANWrapper

from AstarAgent import AstarAgent





import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import os

class GANLevelEnv(gym.Env):
    def __init__(self, gan_model=GANWrapper(), level_dim=32, worker_id=0):
        super().__init__()
        self.step_max_count = 11

        self.gan = gan_model
        self.level_dim = level_dim

        # self.worker_id = worker_id
        self.worker_id = 0

        self.observation = None
        self.segment_buffer = deque(maxlen=self.step_max_count)

        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.level_dim + 3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.level_dim,), dtype=np.float32)

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.level_dim,), dtype=np.float32)  # input vector for GAN

        self.segment_count = 0

        self.log_dir = "./ExperimentLogs/"
        self.playable_file = os.path.join(self.log_dir, "playable_results.txt")
        self.enemy_count_file = os.path.join(self.log_dir, "enemy_count.txt")
        self.reward_file = os.path.join(self.log_dir, "reward.txt")

        os.makedirs(self.log_dir, exist_ok=True)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.segment_buffer.clear()
        self.segment_count = 0
        self.observation = None     
        self.total_score = 0

        self.left_count = 0
        self.middle_count = 0
        self.right_count = 0
        self.score_count = 0

        while self.observation is None:
            self.current_action = np.random.uniform(-1, 1, size=self.level_dim)
            self.current_segment = self.gan.generate(self.current_action, self.worker_id)

            # self.playable = self.is_playable(self.current_segment)
            self.playable = True

            if self.playable:
                side_info = self.segment_side((len(self.segment_buffer) - 1))
                
                # self.observation = np.concatenate([self.current_action, side_info])
                self.observation = self.current_action

                # print("Observation " + str(self.observation) + " added")
                # print("")

        self.segment_buffer.append(self.current_segment)

        info = {"segment": self.current_segment,
                "full_level": self.segment_buffer}
        
        return self.observation, info

    def step(self, action):
        # print("Action mean:", np.mean(action), "std:", np.std(action))
        self.current_action = action
        terminated = False
        truncated = False

        if self.playable and self.segment_count < self.step_max_count:
            self.current_segment = self.gan.generate(self.current_action, self.worker_id)

            # print(str(self.worker_id) + ", self.current_segment: " + str(self.current_segment))

            self.segment_buffer.append(self.current_segment)

            if self.current_segment is None or len(self.current_segment) == 0:
                terminated = True
                reward = -1.0

                side_info = self.segment_side((len(self.segment_buffer) - 1))

                # self.observation = np.concatenate([self.current_action, side_info])
                self.observation = self.current_action

                return self.observation, reward, terminated, False, info

            # self.playable = self.is_playable(self.segment_buffer)
            self.playable = True

            # print("SEGMENT COUNT: " + str(self.segment_count))
            # score, left_count, middle_count, right_count = self.reward(self.segment_buffer)
            score, left, middle, right = self.reward(self.segment_buffer, self.current_segment, (len(self.segment_buffer) - 1))

            self.left_count += left
            self.middle_count += middle
            self.right_count += right
            self.score_count += score

            side_info = self.segment_side((len(self.segment_buffer) - 1))

            # self.observation = np.concatenate([self.current_action, side_info])
            self.observation = self.current_action            

        info = {"segment": self.current_segment,
                "full_level": self.segment_buffer}
        self.segment_count += 1
        
        if not self.playable or self.segment_count >= self.step_max_count:
            with open(self.playable_file, "a", encoding="utf-8") as f:
                f.write(str(self.playable) + ": " + str(self.segment_count) + "\n")

            with open(self.enemy_count_file, "a", encoding="utf-8") as f:
                f.write(str(self.left_count) + " : " + str(self.middle_count) + " : " + str(self.right_count) + "\n")

            terminated = True

            if not self.playable:
                print("NOT PLAYABLE WITH " + str(self.segment_count) + " SEGMENTS")

                score = score - ((11 - self.segment_count) + 1)
                self.score_count = self.score_count - ((11 - self.segment_count) + 1)

                # score = 0

            else:
                print("PLAYABLE WITH " + str(self.segment_count) + " SEGMENTS")

            print("LEVEL SCORE: " + str(score))
            print("==============================")
            print("==============================")

        self.total_score += score

        if terminated == True:
            with open(self.reward_file, "a", encoding="utf-8") as f:
                f.write(str(self.score_count) + "\n")

        return self.observation, score, terminated, truncated, info

    def segment_side(self, segment_idx):
        if segment_idx <= 2:
            return np.array([1, 0, 0], dtype=np.float32)  # left
        elif segment_idx <= 7:
            return np.array([0, 1, 0], dtype=np.float32)  # middle
        else:
            return np.array([0, 0, 1], dtype=np.float32)  # right


    def is_playable(self, segment):
        agent = AstarAgent() 
        playable, playable_distance = agent.AStarRun(segment)

        return playable

    def reward(self, segment_list, segment, index): 
        left = 0
        middle = 0
        right = 0

        # num_enemies = sum(np.count_nonzero(arr == 5) for arr in segment_list)
        num_enemies = sum(np.count_nonzero(arr == 5) for arr in segment)
        if index <= 2:
            weight = 1
            left = num_enemies
        elif index <= 7:
            weight = -2
            middle = num_enemies
        else:
            weight = 1
            right = num_enemies

        # reward = 0.2 * num_enemies + weight * num_enemies
        reward = weight * num_enemies
        # reward = num_enemies

        # num_enemies = sum(np.count_nonzero(arr == 5) for arr in segment)

        return reward, left, middle, right
    
    def _get_observation_stack(self):
        stacked = np.stack(self.observation, axis=0).flatten()
        return stacked.astype(np.float32)

















































# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# from collections import deque

# from GANWrapper import GANWrapper

# from AstarAgent import AstarAgent





# import numpy as np
# from collections import deque
# import matplotlib.pyplot as plt

# import os

# class GANLevelEnv(gym.Env):
#     def __init__(self, gan_model=GANWrapper(), level_dim=32):
#         super().__init__()
#         self.step_max_count = 11

#         self.gan = gan_model
#         self.level_dim = level_dim

#         self.observation = None
#         self.segment_buffer = deque(maxlen=self.step_max_count)

#         self.observation_space = spaces.Box(low=-1, high=1, shape=(self.level_dim + 1 + 3 + 1,), dtype=np.float32)
#         self.action_space = spaces.Box(low=-1, high=1, shape=(self.level_dim,), dtype=np.float32)  # input vector for GAN

#         self.segment_count = 0

#         self.log_dir = "./ExperimentLogs/"
#         self.playable_file = os.path.join(self.log_dir, "playable_results.txt")
#         self.enemy_count_file = os.path.join(self.log_dir, "enemy_count.txt")
#         self.reward_file = os.path.join(self.log_dir, "reward.txt")

#         os.makedirs(self.log_dir, exist_ok=True)

#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)

#         self.segment_buffer.clear()
#         self.segment_count = 0
#         self.observation = None     
#         self.total_score = 0

#         self.left_count = 0
#         self.middle_count = 0
#         self.right_count = 0

#         while self.observation is None:
#             self.current_action = np.random.uniform(-1, 1, size=self.level_dim)
#             self.current_segment = self.gan.generate(self.current_action)

#             # self.playable = self.is_playable(self.current_segment)
#             self.playable = True

#             if self.playable:
#                 self.segment_count += 1

#                 score, left, middle, right = self.reward(self.current_segment, self.segment_count)

#                 total_enemies = left + middle + right
                
#                 if total_enemies != 0:
#                     left_ratio = left / total_enemies
#                     middle_ratio = middle / total_enemies
#                     right_ratio = right / total_enemies

#                     total_enemies = total_enemies  / 50
#                 else:
#                     left_ratio = 0
#                     middle_ratio = 0
#                     right_ratio = 0

#                 progress = np.array([self.segment_count / self.step_max_count], dtype=np.float32)
#                 self.observation = np.concatenate([ self.current_action.astype(np.float32), 
#                                                     progress.astype(np.float32), 
#                                                     np.array([ total_enemies, left_ratio, middle_ratio, right_ratio], 
#                                                     dtype=np.float32)]) 

#                 # self.observation = self.current_action
#                 # print("Observation " + str(self.observation) + " added")
#                 # print("")

#         self.segment_buffer.append(self.current_segment)

#         info = {"segment": self.current_segment,
#                 "full_level": self.segment_buffer}

#         return self.observation, info

#     def step(self, action):
#         # print("Action mean:", np.mean(action), "std:", np.std(action))
#         self.current_action = action
#         terminated = False
#         truncated = False

#         if self.playable and self.segment_count < self.step_max_count:
#             self.current_segment = self.gan.generate(self.current_action)
#             self.segment_buffer.append(self.current_segment)

#             # self.playable = self.is_playable(self.segment_buffer)
#             self.playable = True

#             # print("SEGMENT COUNT: " + str(self.segment_count))
#             # score, left_count, middle_count, right_count = self.reward(self.segment_buffer)
#             score, left, middle, right = self.reward(self.current_segment, self.segment_count)

#             self.left_count += left
#             self.middle_count += middle
#             self.right_count += right

#             # self.observation = self.current_action 
#             total_enemies = left + middle + right

#             if total_enemies != 0:
#                 left_ratio = left / total_enemies
#                 middle_ratio = middle / total_enemies
#                 right_ratio = right / total_enemies

#                 total_enemies = total_enemies  / 50
#             else:
#                 left_ratio = 0
#                 middle_ratio = 0
#                 right_ratio = 0

#             progress = np.array([self.segment_count / self.step_max_count], dtype=np.float32)
#             # self.observation = np.concatenate([self.current_action, progress, [left_ratio, middle_ratio, right_ratio]])  
#             self.observation = np.concatenate([ self.current_action.astype(np.float32), 
#                                                     progress.astype(np.float32), 
#                                                     np.array([ total_enemies, left_ratio, middle_ratio, right_ratio], 
#                                                     dtype=np.float32)])          

#         info = {"segment": self.current_segment,
#                 "full_level": self.segment_buffer}
        
#         self.segment_count += 1
        
#         if not self.playable or self.segment_count >= self.step_max_count:
#             with open(self.playable_file, "a", encoding="utf-8") as f:
#                 f.write(str(self.playable) + ": " + str(self.segment_count) + "\n")

#             with open(self.enemy_count_file, "a", encoding="utf-8") as f:
#                 f.write(str(self.left_count) + " : " + str(self.middle_count) + " : " + str(self.right_count) + "\n")

#             terminated = True

#             if not self.playable:
#                 print("NOT PLAYABLE WITH " + str(self.segment_count) + " SEGMENTS")

#                 # score = -5 * ((11 - self.segment_count) + 1)
#                 score = 0

#             else:
#                 print("PLAYABLE WITH " + str(self.segment_count) + " SEGMENTS")

#             print("LEVEL SCORE: " + str(score))
#             print("==============================")
#             print("==============================")

#         self.total_score += score

#         if terminated == True:
#             with open(self.reward_file, "a", encoding="utf-8") as f:
#                 f.write(str(self.total_score) + "\n")
        
#         print("final: obs: " + str(self.observation) + ", score: " + str(score))
#         return self.observation, score, terminated, truncated, info

#     def is_playable(self, segment):
#         agent = AstarAgent() 
#         playable, playable_distance = agent.AStarRun(segment)

#         return playable

#     def reward(self, segment, index):
#         enemy_count = sum(np.count_nonzero(arr == 5) for arr in segment)

#         if index < 3 or index >= 8:
#             reward = enemy_count
#         else:
#             reward = -0.3 * enemy_count  # softer penalty

#         reward /= 40.0
#         return reward, enemy_count, enemy_count, enemy_count

    
#     def _get_observation_stack(self):
#         stacked = np.stack(self.observation, axis=0).flatten()
#         return stacked.astype(np.float32)
