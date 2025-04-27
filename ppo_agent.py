# ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mean = self.actor(state_tensor)
        dist = Normal(mean, torch.ones_like(mean) * 0.1)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy()[0], log_prob.detach()

    def compute_returns(self, rewards, next_value):
        R = next_value
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)

    def update(self, trajectories):
        states, actions, rewards, old_log_probs = map(torch.tensor, zip(*trajectories))
        returns = self.compute_returns(rewards.tolist(), self.critic(states[-1].unsqueeze(0)).item())
        returns = returns.detach()

        for _ in range(10):
            mean = self.actor(states.float())
            dist = Normal(mean, torch.ones_like(mean) * 0.1)
            log_probs = dist.log_prob(actions.unsqueeze(-1)).sum(dim=-1)

            values = self.critic(states.float()).squeeze()
            advantages = returns - values.detach()

            ratios = torch.exp(log_probs - old_log_probs)
            obj1 = ratios * advantages
            obj2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            loss_actor = -torch.min(obj1, obj2).mean()

            loss_critic = nn.MSELoss()(values, returns)

            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            self.critic_optimizer.step()
