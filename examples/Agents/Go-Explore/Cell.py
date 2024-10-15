import hashlib


def get_state_hash(state):
    state_string = "_".join(str(e) for e in state)
    state_hash = hashlib.md5(state_string.encode()).hexdigest()
    return state_hash


class Cell:

    def __init__(self, state, trajectory_dict):

        self.key = get_state_hash(state)
        self.trajectory_dict = trajectory_dict

        self.state = state
        self.human_vector = []

        self.score = 0
        self.arousal = 0
        self.arousal_values = []
        self.uncertainty = 0
        self.arousal_reward = -1000
        self.behavior_reward = -1000
        self.blended_reward = -1000

        self.age = 0
        self.visited = 1
        self.final = False

    def get_cell_length(self):
        return len(self.trajectory_dict['state_trajectory'])

    def normalize_r_a(self):
        return self.arousal_reward

    def normalize_r_b(self):
        return self.behavior_reward / len(self.trajectory_dict['behavior_trajectory'])

    def assess_cell(self, weight, normalize_behavior, arousal_function):
        # self.behavior_reward = behavior_function(self.trajectory_dict['score_trajectory'], None)
        self.arousal_reward = arousal_function()
        # self.arousal_reward = 0
        if normalize_behavior:
            self.blended_reward = self.normalize_r_a() * weight + self.normalize_r_b() * (1 - weight)
        else:
            self.blended_reward = self.normalize_r_a() * weight + self.behavior_reward * (1 - weight)
        print(self.arousal_reward, self.blended_reward)

    def update_key(self, state):
        self.key = get_state_hash(state)
