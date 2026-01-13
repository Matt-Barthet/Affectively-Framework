import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from GANWrapper import GANWrapper

from AstarAgent import AstarAgent





import numpy as np
from collections import deque
import matplotlib.pyplot as plt
observation_buffer = deque(maxlen=5)

for i in range(5):
    current_action = np.random.uniform(-1, 1, size=32)
    observation_buffer.append(current_action)
# print("observation_buffer: " + str(observation_buffer))
actions = np.array(observation_buffer)
# print("actions: " + str(actions))

# print("range(len(actions)): " + str(range(len(actions)*32)))
# plt.scatter((range(len(actions)*32)), actions)
# plt.xlabel("Action Number (32 values per action), 5 total action")
# plt.ylabel("Action Value")
# plt.show()





class GANLevelEnv(gym.Env):
    def __init__(self, gan_model=GANWrapper(), level_dim=32):
        super().__init__()
        self.step_max_count = 11

        self.gan = gan_model
        self.level_dim = level_dim

        self.observation_buffer = deque(maxlen=5)
        self.segment_buffer = deque(maxlen=self.step_max_count)

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.level_dim * 5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.level_dim,), dtype=np.float32)  # input vector for GAN

        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.observation_buffer.clear()
        self.segment_buffer.clear()
        self.step_count = 0
        
        # self.current_action = np.random.uniform(-1, 1, size=self.level_dim)
        # self.current_action = np.zeros(self.level_dim) 
        for _ in range(5):
            self.current_action = np.random.uniform(-1, 1, size=self.level_dim)
            # print(str(_) + ": " + str(self.current_action))
            self.observation_buffer.append(self.current_action)
            # self.current_segment = self.gan.generate(self.current_action)
            # self.segment_buffer.append(self.current_segment)

        # actions = np.array(self.observation_buffer)
        # plt.scatter((range(len(actions)*32)), actions)
        # plt.show()
            
        return self._get_observation_stack(), {}

    def step(self, action):
        self.current_segment = 0
        playable = False
        n = 0

        while playable == False:
            self.current_action = np.random.uniform(-1, 1, size=self.level_dim)
            self.current_segment = self.gan.generate(self.current_action)
            playable = self.is_playable(self.current_segment)

        self.segment_buffer.append(self.current_segment)

        while playable == True and n < 11:
            action, _ = model.predict(obs, deterministic=True)
            self.current_segment = self.gan.generate(action)
            playable = self.is_playable(self.current_segment)
            score = self.reward(self.segment_buffer)

        self.current_action = action
        self.observation_buffer.append(self.current_action)

        is_valid = np.all((self.current_action >= -1) & (self.current_action <= 1))
        # if not is_valid:
        #     print(f"WONG ACTION STEP: {self.current_action}")
        
        self.current_segment = self.gan.generate(self.current_action)
        reward = self.evaluate_level(self.segment_buffer)

        self.segment_buffer.append(self.current_segment)

        # self.step_count += 1

        # reward = self.evaluate_level(self.segment_buffer)
        # print("Reward: " + str(reward))

        reward = 0
        terminated = False
        self.step_count += 1
        if self.step_count < self.step_max_count:
            reward = 0
            terminated = False
        else:
            reward = self.evaluate_level(self.segment_buffer)
            terminated = True
            self.step_count = 0

        truncated = False
        info = {"segment": self.current_segment,
                "full_level": self.segment_buffer}
        # info = {"segment": self.current_segment}
        
        return self._get_observation_stack(), reward, terminated, truncated, info

    def evaluate_level(self, segments):
        # print("segments:")
        # print(str(segments))

        # left_num_enemies = 0
        # middle_num_enemies = 0
        # right_num_enemies = 0
        # for i in range(11):
        #     if i <= 2:
        #         left_num_enemies += sum(np.count_nonzero(arr == 5) for arr in segments[i])
        #     elif i > 2 and i <= 7:
        #         middle_num_enemies += sum(np.count_nonzero(arr == 5) for arr in segments[i])
        #     else:
        #         right_num_enemies += sum(np.count_nonzero(arr == 5) for arr in segments[i])

        num_enemies = sum(np.count_nonzero(arr == 5) for arr in segments)
        num_special_blocks = sum(np.count_nonzero((arr == 3) | (arr == 4)) for arr in segments)

        agent = AstarAgent() 
        playability, playable_distance = agent.AStarRun(segments)

        # left_enemy_score = left_num_enemies / 50.0
        # middle_enemy_score = middle_num_enemies / 50.0
        # right_enemy_score = right_num_enemies / 50.0

        distance_score = playable_distance / 170.0

        special_blocks_score = num_special_blocks / 100

        # reward = 0.7 * distance_score + 0.09 * left_enemy_score - 0.15 * middle_enemy_score + 0.09 * right_enemy_score
        # reward = 0.7 * distance_score + 0.3 * (0.3 * left_enemy_score - 0.5 * middle_enemy_score + 0.3 * right_enemy_score)
        # reward = distance_score
        reward = 0.5 * distance_score + 0.5 * special_blocks_score

        if not playability:
            reward -= 1.0

        reward *= 5.0

        print("=====================================")
        print("PLAYABLE: " + str(playability))
        print("PLAYABLE DISTANCE: " + str(playable_distance))
        print("NUMBER OF ENEMIES: " + str(num_enemies))
        print("NUMBER OF SPECIAL BLOCKS: " + str(num_special_blocks))
        # print("NUMBER OF LEFT ENEMIES: " + str(left_num_enemies))
        # print("NUMBER OF MIDDLE ENEMIES: " + str(middle_num_enemies))
        # print("NUMBER OF RIGHT ENEMIES: " + str(right_num_enemies))
        print("REWARD: " + str(reward))
        print("=====================================")
        return reward

































    # def step(self, action):
    #     self.current_action = action
    #     self.observation_buffer.append(self.current_action)

    #     is_valid = np.all((self.current_action >= -1) & (self.current_action <= 1))
    #     # if not is_valid:
    #     #     print(f"WONG ACTION STEP: {self.current_action}")
        
    #     self.current_segment = self.gan.generate(self.current_action)
    #     self.segment_buffer.append(self.current_segment)

    #     # self.step_count += 1

    #     # reward = self.evaluate_level(self.segment_buffer)
    #     # print("Reward: " + str(reward))

    #     reward = 0
    #     terminated = False
    #     self.step_count += 1
    #     if self.step_count < self.step_max_count:
    #         reward = 0
    #         terminated = False
    #     else:
    #         # reward = self.evaluate_level(self.segment_buffer)
    #         terminated = True
    #         self.step_count = 0

    #     truncated = False
    #     info = {"segment": self.current_segment,
    #             "full_level": self.segment_buffer}
    #     # info = {"segment": self.current_segment}
        
    #     return self._get_observation_stack(), reward, terminated, truncated, info

    # def evaluate_level(self, segments):
    #     # print("segments:")
    #     # print(str(segments))

    #     # left_num_enemies = 0
    #     # middle_num_enemies = 0
    #     # right_num_enemies = 0
    #     # for i in range(11):
    #     #     if i <= 2:
    #     #         left_num_enemies += sum(np.count_nonzero(arr == 5) for arr in segments[i])
    #     #     elif i > 2 and i <= 7:
    #     #         middle_num_enemies += sum(np.count_nonzero(arr == 5) for arr in segments[i])
    #     #     else:
    #     #         right_num_enemies += sum(np.count_nonzero(arr == 5) for arr in segments[i])

    #     num_enemies = sum(np.count_nonzero(arr == 5) for arr in segments)
    #     num_special_blocks = sum(np.count_nonzero((arr == 3) | (arr == 4)) for arr in segments)

    #     agent = AstarAgent() 
    #     playability, playable_distance = agent.AStarRun(segments)

    #     # left_enemy_score = left_num_enemies / 50.0
    #     # middle_enemy_score = middle_num_enemies / 50.0
    #     # right_enemy_score = right_num_enemies / 50.0

    #     distance_score = playable_distance / 170.0

    #     special_blocks_score = num_special_blocks / 100

    #     # reward = 0.7 * distance_score + 0.09 * left_enemy_score - 0.15 * middle_enemy_score + 0.09 * right_enemy_score
    #     # reward = 0.7 * distance_score + 0.3 * (0.3 * left_enemy_score - 0.5 * middle_enemy_score + 0.3 * right_enemy_score)
    #     # reward = distance_score
    #     reward = 0.5 * distance_score + 0.5 * special_blocks_score

    #     if not playability:
    #         reward -= 1.0

    #     reward *= 5.0

    #     print("=====================================")
    #     print("PLAYABLE: " + str(playability))
    #     print("PLAYABLE DISTANCE: " + str(playable_distance))
    #     print("NUMBER OF ENEMIES: " + str(num_enemies))
    #     print("NUMBER OF SPECIAL BLOCKS: " + str(num_special_blocks))
    #     # print("NUMBER OF LEFT ENEMIES: " + str(left_num_enemies))
    #     # print("NUMBER OF MIDDLE ENEMIES: " + str(middle_num_enemies))
    #     # print("NUMBER OF RIGHT ENEMIES: " + str(right_num_enemies))
    #     print("REWARD: " + str(reward))
    #     print("=====================================")
    #     return reward













































    # def evaluate_level(self, segments):
    #     print("segments:")
    #     print(str(segments))

    #     left_num_enemies = 0
    #     middle_num_enemies = 0
    #     right_num_enemies = 0
    #     for i in range(11):
    #         # print(str(i) + "segment: " + str(segments[i]))
    #         # print(str(i) + "enemy count: " + str(sum(np.count_nonzero(arr == 5) for arr in segments[i])))
    #         # print("--------")

    #         if i <= 3:
    #             left_num_enemies += sum(np.count_nonzero(arr == 5) for arr in segments[i])
    #             print(str(i) + ", left_num_enemies: " + str(left_num_enemies))
    #         elif i > 3 and i <= 8:
    #             middle_num_enemies += sum(np.count_nonzero(arr == 5) for arr in segments[i])
    #             print(str(i) + ", middle_num_enemies: " + str(middle_num_enemies))
    #         else:
    #             right_num_enemies += sum(np.count_nonzero(arr == 5) for arr in segments[i])
    #             print(str(i) + ", right_num_enemies: " + str(right_num_enemies))

    #     num_enemies = sum(np.count_nonzero(arr == 5) for arr in segments)
    #     # print("num_enemies: " + str(num_enemies))

    #     # print(str(segments))
    #     # num_enemies = np.sum(observation == 5)
    #     # return num_enemies



    #     # reward = 0
    #     # return reward
    #     # -----
    #     agent = AstarAgent() 
    #     playability, playable_distance = agent.AStarRun(segments)

    #     # if playability:
    #     #     reward = (6 * num_enemies) + playable_distance
    #     # else:
    #     #     reward = (playable_distance) / 3


    #     # if playability:
    #     #     reward = (6 * num_enemies) + playable_distance
    #     # else:
    #     #     reward = 0

    #     # 5 4000 ep 10 mul enemy only if playable
    #     # if playability:
    #     #     reward = 10 * num_enemies
    #     # else:
    #     #     reward = 0

    #     # 6 4000 ep reward improvement greater enemy reward harsher fail
    #     # if playability:
    #     #     reward = ((10 * num_enemies) + playable_distance) * 2
    #     # else:
    #     #     reward = ((10 * num_enemies) + playable_distance) / 2

    #     enemy_score = num_enemies / 50.0
    #     distance_score = playable_distance / 170.0

    #     reward = 0.7 * distance_score + 0.3 * enemy_score

    #     if not playability:
    #         reward -= 1.0

    #     reward *= 5.0


        

    #     # reward = playability
    #     # 160 25
    #     print("=====================================")
    #     print("PLAYABLE: " + str(playability))
    #     print("PLAYABLE DISTANCE: " + str(playable_distance))
    #     print("NUMBER OF ENEMIES: " + str(num_enemies))
    #     print("REWARD: " + str(reward))
    #     print("=====================================")
    #     return reward
    #     # -----





    #     # target = 300
        
    #     # reward = 1 - abs(length - target) / target
    #     # reward = max(reward, -1) 

    #     # return reward
    
    def _get_observation_stack(self):
        stacked = np.stack(self.observation_buffer, axis=0).flatten()
        return stacked.astype(np.float32)
