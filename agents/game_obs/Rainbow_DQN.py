import math
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            nn.ReLU()
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


class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha  # Controls the level of prioritization (0 - uniform, 1 - full prioritization)
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

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
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

        batch = list(zip(*samples))
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32, device=device)
        actions = np.stack(batch[1], axis=0).astype(np.int64)
        rewards = torch.tensor(batch[2], dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32, device=device)
        dones = torch.tensor(batch[4], dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class RainbowAgent:
    def __init__(self, observation_size, action_sizes, atom_size=51, v_min=-10, v_max=10,
                 n_step=3, gamma=0.99, lr=1e-4, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.action_sizes = action_sizes
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.n_step = n_step
        self.gamma = gamma

        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(device)
        self.delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)

        self.policy_net = MultiDiscreteRainbowDQN(observation_size, action_sizes, atom_size, self.support).to(device)
        self.target_net = MultiDiscreteRainbowDQN(observation_size, action_sizes, atom_size, self.support).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory = PrioritizedReplayBuffer(100000, alpha)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_idx = 0

        self.n_step_buffer = deque(maxlen=n_step)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.policy_net.reset_noise()
        with torch.no_grad():
            q_values, _ = self.policy_net(state)
        action = [qv.argmax(1).item() for qv in q_values]
        return action

    def append_sample(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        reward, next_state, done = self._get_n_step_info()
        state, action = self.n_step_buffer[0][:2]

        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        self.memory.push(state, action, reward, next_state, done)

    def _get_n_step_info(self):
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_s, d = transition[2:]
            reward = r + self.gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)
        return reward, next_state, done

    def compute_td_loss(self, batch_size):
        beta = min(1.0, self.beta_start + self.frame_idx * (1.0 - self.beta_start) / self.beta_frames)
        self.frame_idx += 1

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size, beta)

        losses = []
        for dim in range(len(self.action_sizes)):
            actions_dim = torch.tensor(actions[:, dim], dtype=torch.long, device=device).unsqueeze(1)

            with torch.no_grad():
                next_q_values, next_q_distributions = self.target_net(next_states)
                next_actions = next_q_values[dim].argmax(1)
                next_q_distribution = next_q_distributions[dim][range(batch_size), next_actions]

                t_z = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (
                            self.gamma ** self.n_step) * self.support.unsqueeze(0)
                t_z = t_z.clamp(self.v_min, self.v_max)
                b = (t_z - self.v_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()

                offset = (torch.arange(0, batch_size) * self.atom_size).unsqueeze(1).to(device)
                m = torch.zeros(batch_size, self.atom_size, device=device)
                m.view(-1).index_add_(0, (l + offset).view(-1), (next_q_distribution * (u.float() - b)).view(-1))
                m.view(-1).index_add_(0, (u + offset).view(-1), (next_q_distribution * (b - l.float())).view(-1))

            q_values, q_distributions = self.policy_net(states)
            actions_dim_expanded = actions_dim.unsqueeze(2).expand(batch_size, 1, self.atom_size)
            q_distribution = q_distributions[dim].gather(1, actions_dim_expanded).squeeze(1)

            loss = - (m * torch.log(q_distribution + 1e-8)).sum(1)  # Shape: [batch_size]
            losses.append(loss)

        loss = sum(losses)
        loss = loss * weights
        loss_mean = loss.mean()
        priorities = loss.detach().cpu().numpy() + 1e-6

        self.memory.update_priorities(indices, priorities)

        self.optimizer.zero_grad()
        loss_mean.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        return loss_mean.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'frame_idx': self.frame_idx,
        }, os.path.join(save_path, 'agent_checkpoint.pth'))
        # print(f"Agent saved to {save_path}")

    def load(self, load_path):
        checkpoint = torch.load(os.path.join(load_path, 'agent_checkpoint.pth'), map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.frame_idx = checkpoint.get('frame_idx', 0)
        # print(f"Agent loaded from {load_path}")


# In the training loop
def train(agent, env, num_episodes=500, batch_size=64, update_target_every=1000, learning_starts=1000, name=""):
    total_steps = 0
    training_started = False
    episode_rewards = []

    # Wrap the episodes loop with tqdm
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        state = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0

        for _ in tqdm(range(600), desc=f"Episode {episode+1}/{num_episodes} Steps", leave=False):
        # for _ in range(600):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32)
            agent.append_sample(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(agent.memory) >= learning_starts:
                if not training_started:
                    training_started = True
                _ = agent.compute_td_loss(batch_size)

            if total_steps % update_target_every == 0 and training_started:
                agent.update_target()

        episode_rewards.append(episode_reward)

        if episode % 100 == 0:
            agent.save(f'./Results/DQN/DQN_{name}_Checkpoint')

    print("Training completed.")

# TODO: Fix this script to work with updated framework
def main(run, weight, env_type):
    np.set_printoptions(suppress=True, precision=6)

    preference_task = True
    classification_task = False

    sideChannel = env.customSideChannel
    env.targetSignal = np.ones

    if weight == 0:
        label = 'optimize'
    elif weight == 0.5:
        label = 'blended'
    else:
        label = 'arousal'

    agent = RainbowResnetAgent(env.observation_space.shape[0], env.action_space.nvec.tolist())
    num_episodes = 16638
    batch_size = 64
    update_target_every = 600
    train(agent, env, num_episodes, batch_size, update_target_every, name=f"run{run}")
