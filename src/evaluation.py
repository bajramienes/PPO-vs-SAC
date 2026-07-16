import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Set consistent colors ===
ppo_color = '#4CAF50'    # Green for PPO
sac_color = '#FFEB3B'    # Yellow for SAC
baseline_color = '#9E9E9E'  # Gray for Rule-Based

# === Output folder ===
output_dir = "NEW CHARTS"
os.makedirs(output_dir, exist_ok=True)

# Simulated Energy Environment
class EnergyEnvironment:
    def __init__(self, data):
        self.data = (data - data.min()) / (data.max() - data.min())
        self.state_dim = 6
        self.action_dim = 1
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.data.iloc[0].values

    def step(self, action, use_rule_based=False):
        self.current_step += 1
        state = self.data.iloc[self.current_step % len(self.data)].values
        if use_rule_based:
            reward = 0.5
        else:
            efficiency = 1 - abs(action - state[0])
            emissions = 1 - state[1] * action
            stability = 1 - abs(action - 0.5)
            reward = 0.4 * efficiency + 0.4 * emissions + 0.2 * stability
        done = self.current_step >= self.max_episodes
        return state, reward, done

# PPO, SAC, Rule-Based
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(),
                                    nn.Linear(32, action_dim), nn.Tanh())
    def act(self, state):
        return np.clip(self.policy(torch.FloatTensor(state)).detach().numpy(), 0, 1)

class SAC:
    def __init__(self, state_dim, action_dim):
        self.policy = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(),
                                    nn.Linear(32, action_dim), nn.Tanh())
    def act(self, state):
        return np.clip(self.policy(torch.FloatTensor(state)).detach().numpy() + np.random.normal(0, 0.1), 0, 1)

class RuleBased:
    def act(self, state):
        return 1.0 if state[0] > 0.7 else 0.3

# Load dataset
data = pd.read_csv(r'C:\Users\Enes\Desktop\PPO vs SAC\World Energy Consumption.csv')
indicators = ['electricity_generation', 'greenhouse_gas_emissions', 'renewables_share_energy',
              'fossil_share_energy', 'solar_energy_per_capita', 'oil_consumption']
data = data[indicators].dropna()

# Initialize environment and agents
env = EnergyEnvironment(data)
agents = {'Rule-Based': RuleBased(), 'PPO': PPO(6, 1), 'SAC': SAC(6, 1)}
episode_counts = [300, 500, 700, 1000, 1500, 2000, 2500, 3000]
metrics = {name: [] for name in agents}

# Run experiments
for num_episodes in episode_counts:
    env.max_episodes = num_episodes
    print(f"Running {num_episodes} episodes...")
    for name, agent in agents.items():
        rewards, actions = [], []
        state = env.reset()
        import time; start = time.time()
        for _ in range(num_episodes):
            action = agent.act(state)
            next_state, reward, done = env.step(action, use_rule_based=(name == 'Rule-Based'))
            rewards.append(float(reward))
            actions.append(float(action))
            state = next_state
        elapsed = time.time() - start
        # Safeguards
        total_reward = np.sum(rewards) if rewards else 0.0
        speed = num_episodes / elapsed if elapsed > 0 else 0.0
        variance = np.var(actions) if len(actions) > 1 else 0.0
        avg_action = np.mean(actions) if len(actions) > 0 else 0.0
        metrics[name].append({
            'Reward': total_reward,
            'Speed': speed,
            'Variance': variance,
            'AvgAction': avg_action
        })

# === Generate Bar Charts ===
def bar_chart(metric_key, ylabel, filename):
    avg_values = {
        'PPO': [m[metric_key] for m in metrics['PPO']],
        'SAC': [m[metric_key] for m in metrics['SAC']],
        'Rule-Based': [m[metric_key] for m in metrics['Rule-Based']]
    }
    x = np.arange(len(episode_counts))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, avg_values['PPO'], width, label='PPO', color=ppo_color)
    ax.bar(x, avg_values['SAC'], width, label='SAC', color=sac_color)
    ax.bar(x + width, avg_values['Rule-Based'], width, label='Rule-Based', color=baseline_color)
    ax.set_xticks(x)
    ax.set_xticklabels(episode_counts)
    ax.set_xlabel('Episode Count')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} Across Episode Counts')
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close()

bar_chart('Variance', 'Action Variance', 'action_variance_updated.pdf')
bar_chart('Speed', 'Execution Speed (Episodes per Second)', 'algorithm_speed_updated.pdf')
bar_chart('AvgAction', 'Average Action Score', 'average_action_score_updated.pdf')
bar_chart('Reward', 'Cumulative Reward (Balkan vs Nordic)', 'ppo_vs_sac_comparison_updated.pdf')

# === Generate Radar Chart ===
def radar_chart(filename):
    import math
    labels = ['AvgAction', 'Stability', 'Speed']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    def normalize(data):
        min_val, max_val = np.min(data), np.max(data)
        return (data - min_val) / (max_val - min_val) if max_val > min_val else np.ones_like(data)

    ppo_avg = [np.mean([m['AvgAction'] for m in metrics['PPO']]),
               1 / np.mean([m['Variance'] for m in metrics['PPO']]) if np.mean([m['Variance'] for m in metrics['PPO']]) > 0 else 0.0,
               np.mean([m['Speed'] for m in metrics['PPO']])]
    sac_avg = [np.mean([m['AvgAction'] for m in metrics['SAC']]),
               1 / np.mean([m['Variance'] for m in metrics['SAC']]) if np.mean([m['Variance'] for m in metrics['SAC']]) > 0 else 0.0,
               np.mean([m['Speed'] for m in metrics['SAC']])]

    ppo_norm = normalize(np.array(ppo_avg))
    sac_norm = normalize(np.array(sac_avg))

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, np.append(ppo_norm, ppo_norm[0]), color=ppo_color, linewidth=2, label='PPO')
    ax.fill(angles, np.append(ppo_norm, ppo_norm[0]), color=ppo_color, alpha=0.25)
    ax.plot(angles, np.append(sac_norm, sac_norm[0]), color=sac_color, linewidth=2, label='SAC')
    ax.fill(angles, np.append(sac_norm, sac_norm[0]), color=sac_color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('PPO vs SAC: Combined Metrics (Normalized)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
    plt.close()

radar_chart('ppo_vs_sac_radar_summary_updated.pdf')
print("All updated charts saved in NEW CHARTS")
