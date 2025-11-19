import os
from collections import deque

import numpy as np
import torch
from torch import optim
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from examples.Agents.DQN.Rainbow_DQN import NoisyLinear, PrioritizedReplayBuffer, device


class MultiDiscreteRainbowResNetDQN(nn.Module):
    def __init__(self, input_channels, action_sizes, atom_size, support):
        super(MultiDiscreteRainbowResNetDQN, self).__init__()
        self.input_channels = input_channels
        self.action_sizes = action_sizes
        self.atom_size = atom_size
        self.support = support

        # Load a pretrained ResNet
        self.resnet = resnet18(pretrained=True)
        # Modify the first layer to match the number of input channels
        if input_channels != 3:  # ResNet expects 3 channels by default
            self.resnet.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
        # Remove the fully connected classification head
        self.resnet.fc = nn.Identity()

        # Compute the ResNet output size
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, 84, 84)  # Example input shape
            resnet_output_dim = self.resnet(sample_input).view(-1).size(0)

        # Define value and advantage streams
        self.value_layers = nn.ModuleList()
        self.advantage_layers = nn.ModuleList()

        for action_size in action_sizes:
            value_layer = nn.Sequential(
                NoisyLinear(resnet_output_dim, 128),
                nn.ReLU(),
                NoisyLinear(128, self.atom_size),
            )
            self.value_layers.append(value_layer)

            advantage_layer = nn.Sequential(
                NoisyLinear(resnet_output_dim, 128),
                nn.ReLU(),
                NoisyLinear(128, action_size * self.atom_size),
            )
            self.advantage_layers.append(advantage_layer)

    def forward(self, x):
        batch_size = x.size(0)

        # Extract features using ResNet
        x = self.resnet(x)
        x = x.view(batch_size, -1)  # Flatten the ResNet output

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


class RainbowResnetAgent:
    def __init__(self, input_channels, action_sizes, atom_size=51, v_min=-10, v_max=10,
                 n_step=3, gamma=0.99, lr=1e-4, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.action_sizes = action_sizes
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.n_step = n_step
        self.gamma = gamma

        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(device)
        self.delta_z = (self.v_max - self.v_min) / (self.atom_size - 1)

        # Update policy and target networks to use ResNet-based DQN
        self.policy_net = MultiDiscreteRainbowResNetDQN(input_channels, action_sizes, atom_size, self.support).to(device)
        self.target_net = MultiDiscreteRainbowResNetDQN(input_channels, action_sizes, atom_size, self.support).to(device)
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
def learn(agent, env, num_episodes=500, batch_size=64, update_target_every=1000, learning_starts=1000, name=""):
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
