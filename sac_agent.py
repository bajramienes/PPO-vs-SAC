# sac_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + output_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class SACAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4):
        self.actor = Actor(obs_dim, act_dim)
        self.critic1 = Critic(obs_dim, act_dim)
        self.critic2 = Critic(obs_dim, act_dim)
        self.target_critic1 = Critic(obs_dim, act_dim)
        self.target_critic2 = Critic(obs_dim, act_dim)

        # Copy weights from critics to target critics
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)

        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -act_dim

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mean = self.actor(state_tensor)
        dist = Normal(mean, torch.ones_like(mean) * 0.1)
        action = dist.sample()
        return action.detach().numpy()[0]

    def update(self, replay_buffer, batch_size=64, gamma=0.99, tau=0.005):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states = replay_buffer.sample(batch_size)

        # Update critics
        with torch.no_grad():
            next_action = self.actor(next_states)
            target_q1 = self.target_critic1(next_states, next_action)
            target_q2 = self.target_critic2(next_states, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards + gamma * target_q.squeeze()

        q1 = self.critic1(states, actions).squeeze()
        q2 = self.critic2(states, actions).squeeze()

        critic1_loss = nn.MSELoss()(q1, target_value)
        critic2_loss = nn.MSELoss()(q2, target_value)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        new_action = self.actor(states)
        q1_new = self.critic1(states, new_action)
        q2_new = self.critic2(states, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = -q_new.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

