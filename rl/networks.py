"""MAPPO Actor-Critic 与 Rollout 存储。"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from LLM4RD import config as cfg


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=cfg.HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state.float()))
        return F.relu(self.fc2(x))

    def get_action_probs(self, state):
        features = self.forward(state)
        return F.softmax(self.actor_head(features), dim=-1)

    def get_value(self, state):
        return self.critic_head(self.forward(state))

    def evaluate_actions(self, state, action):
        features = self.forward(state)
        dist = Categorical(logits=self.actor_head(features))
        return self.critic_head(features), dist.log_prob(action), dist.entropy().mean()


class RolloutStorage:
    def __init__(self, num_steps, num_agents, state_dim, device):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.device = device
        self.states = torch.zeros((num_steps + 1, num_agents, state_dim)).to(device)
        self.rewards = torch.zeros((num_steps, num_agents, 1)).to(device)
        self.actions = torch.zeros((num_steps, num_agents, 1)).long().to(device)
        self.log_probs = torch.zeros((num_steps, num_agents, 1)).to(device)
        self.values = torch.zeros((num_steps + 1, num_agents, 1)).to(device)
        self.masks = torch.ones((num_steps + 1, num_agents, 1)).to(device)
        self.returns = torch.zeros((num_steps + 1, num_agents, 1)).to(device)
        self.advantages = torch.zeros((num_steps, num_agents, 1)).to(device)
        self.step = 0

    def insert(self, state, action, log_prob, value, reward, mask):
        self.states[self.step + 1].copy_(state)
        self.actions[self.step].copy_(action)
        self.log_probs[self.step].copy_(log_prob)
        self.values[self.step].copy_(value)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.step = 0

    def compute_returns_and_advantages(self, next_value, gamma, gae_lambda):
        self.values[-1] = next_value
        gae = 0
        for step in reversed(range(self.num_steps)):
            delta = (
                self.rewards[step]
                + gamma * self.values[step + 1] * self.masks[step + 1]
                - self.values[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.values[step]
            self.advantages[step] = gae

    def feed_forward_generator(self, advantages, num_mini_batch):
        batch_size = self.num_agents * self.num_steps
        mini_batch_size = max(1, batch_size // max(1, num_mini_batch))
        sampler = torch.randperm(batch_size)
        states = self.states[:-1].view(-1, self.states.size(-1))
        actions = self.actions.view(-1, 1)
        log_probs = self.log_probs.view(-1, 1)
        returns = self.returns[:-1].view(-1, 1)
        advantages = advantages.view(-1, 1)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            indices = sampler[start:end]
            yield (
                states[indices],
                actions[indices].squeeze(-1),
                log_probs[indices],
                returns[indices],
                advantages[indices],
            )
