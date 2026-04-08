import torch.nn as nn
import torch
import math 
import torch.nn.functional as F
import numpy as np

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
    

class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha, device):
        self.capacity = capacity
        self.alpha = alpha  # Controls the level of prioritization (0 - uniform, 1 - full prioritization)
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.device=device

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        action = np.array(action, dtype=np.int64)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        batch = list(zip(*samples))
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32, device=self.device)
        actions = np.stack(batch[1], axis=0).astype(np.int64)
        rewards = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch[4], dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)
    
class MultiDiscreteRainbowDQN(nn.Module):
    def __init__(self, observation_size, action_sizes, atom_size, support):
        super(MultiDiscreteRainbowDQN, self).__init__()
        self.observation_size = observation_size
        self.action_sizes = action_sizes
        self.atom_size = atom_size
        self.support = support

        self.feature_layer = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.value_layers = nn.ModuleList()
        self.advantage_layers = nn.ModuleList()

        for action_size in action_sizes:
            value_layer = nn.Sequential(
                NoisyLinear(128, 128),
                nn.ReLU(),
                NoisyLinear(128, self.atom_size)
            )
            self.value_layers.append(value_layer)

            advantage_layer = nn.Sequential(
                NoisyLinear(128, 128),
                nn.ReLU(),
                NoisyLinear(128, action_size * self.atom_size)
            )
            self.advantage_layers.append(advantage_layer)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_layer(x)

        q_values = []
        q_distributions = []

        for value_layer, advantage_layer, action_size in zip(
            self.value_layers, self.advantage_layers, self.action_sizes
        ):
            value = value_layer(x).view(batch_size, 1, self.atom_size)
            advantage = advantage_layer(x).view(batch_size, action_size, self.atom_size)
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
            q_distribution = F.softmax(q_atoms, dim=2)  # Distribution over atoms
            q_value = torch.sum(q_distribution * self.support, dim=2)
            q_values.append(q_value)
            q_distributions.append(q_distribution)

        return q_values, q_distributions

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()