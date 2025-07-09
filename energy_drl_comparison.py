import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# === Create new folder for charts ===
output_dir = "NEW CHARTS"
os.makedirs(output_dir, exist_ok=True)

# Simulated energy environment
class EnergyEnvironment:
    def __init__(self, data):
        self.data = data
        self.state_dim = 6  # Six energy indicators
        self.action_dim = 1  # Action: energy allocation (0 to 1)
        self.max_episodes = 300
        self.current_step = 0
        # Normalize data to [0, 1]
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())

    def reset(self):
        self.current_step = 0
        return self.data.iloc[0].values

    def step(self, action, use_rule_based=False):
        self.current_step += 1
        state = self.data.iloc[self.current_step % len(self.data)].values
        if use_rule_based:
            reward = 0.5  # Constant reward for Rule-Based
        else:
            efficiency = 1 - abs(action - state[0])
            emissions = 1 - state[1] * action
            stability = 1 - abs(action - 0.5)
            reward = 0.4 * efficiency + 0.4 * emissions + 0.2 * stability
        done = self.current_step >= self.max_episodes
        return state, reward, done

# PPO implementation
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(),
                                    nn.Linear(32, action_dim), nn.Tanh())
        self.epsilon = 0.2
        self.actions, self.rewards = [], []

    def act(self, state):
        state = torch.FloatTensor(state)
        action = self.policy(state).detach().numpy()
        action = np.clip(action, 0, 1)
        return action

    def store(self, action, reward):
        self.actions.append(action)
        self.rewards.append(reward)

# SAC implementation
class SAC:
    def __init__(self, state_dim, action_dim):
        self.policy = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(),
                                    nn.Linear(32, action_dim), nn.Tanh())
        self.alpha = 0.2
        self.actions, self.rewards = [], []

    def act(self, state):
        state = torch.FloatTensor(state)
        action = self.policy(state).detach().numpy()
        action = np.clip(action + np.random.normal(0, 0.1), 0, 1)
        return action

    def store(self, action, reward):
        self.actions.append(action)
        self.rewards.append(reward)

# Rule-based baseline
class RuleBased:
    def __init__(self):
        self.actions, self.rewards = [], []

    def act(self, state):
        demand = state[0]
        action = 1.0 if demand > 0.7 else 0.3
        return action

    def store(self, action, reward):
        self.actions.append(action)
        self.rewards.append(reward)

# Load and preprocess data
data = pd.read_csv(r'C:\\Users\\Enes\\Desktop\\PPO vs SAC\\World Energy Consumption.csv')
indicators = ['electricity_generation', 'greenhouse_gas_emissions', 'renewables_share_energy',
              'fossil_share_energy', 'solar_energy_per_capita', 'oil_consumption']
data = data[indicators].dropna()

# Initialize environment and agents
env = EnergyEnvironment(data)
rule_based = RuleBased()
ppo = PPO(state_dim=6, action_dim=1)
sac = SAC(state_dim=6, action_dim=1)
agents = {'Rule-Based': rule_based, 'PPO': ppo, 'SAC': sac}
labels = ['Rule-Based', 'PPO', 'SAC']

# === Run for multiple episode counts ===
for num_episodes in [300, 500, 700, 1000, 1500, 2000, 2500, 3000]:
    print(f"Running evaluation for {num_episodes} episodes...")
    env.max_episodes = num_episodes
    metrics = {'Cumulative Reward': [], 'Execution Speed': []}

    for agent_name, agent in agents.items():
        agent.rewards, agent.actions = [], []
        start_time = time.time()
        state = env.reset()
        for episode in range(num_episodes):
            action = agent.act(state)
            use_rule_based = (agent_name == 'Rule-Based')
            next_state, reward, done = env.step(action, use_rule_based=use_rule_based)
            agent.store(action, reward)
            state = next_state
            if done:
                break
        exec_time = time.time() - start_time
        rewards = np.array(agent.rewards)
        metrics['Cumulative Reward'].append(np.sum(rewards))
        metrics['Execution Speed'].append(num_episodes / exec_time)
        print(f"  {agent_name}: Total Reward={np.sum(rewards):.2f}, Speed={num_episodes/exec_time:.2f} eps/s")

    # Plot and save bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, metrics['Cumulative Reward'], width, label='Cumulative Reward', color='#4CAF50')
    ax.bar(x + width/2, metrics['Execution Speed'], width, label='Exec Speed (eps/s)', color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Value (Reward / Episodes per Second)')
    ax.set_title(f'Performance Metrics ({num_episodes} Episodes)')
    ax.legend()
    chart_path = os.path.join(output_dir, f'drl_metrics_{num_episodes}_episodes.pdf')
    plt.savefig(chart_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {chart_path}")

print("All evaluations completed. Results saved in NEW CHARTS folder.")
