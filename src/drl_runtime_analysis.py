import time
import torch
import torch.nn as nn
import numpy as np
import psutil
import GPUtil

# Dummy PPO Model
class PPOModel(nn.Module):
    def __init__(self):
        super(PPOModel, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

    def train_dummy(self, episodes, steps_per_episode):
        total_steps = episodes * steps_per_episode
        for _ in range(total_steps):
            state = torch.rand(1, 6)
            action = self.actor(state)
            value = self.critic(state)
            advantage = torch.rand(1, 1)
            loss = ((action - 0.5) ** 2).mean() + ((value - 0.5) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Dummy SAC Model
class SACModel(nn.Module):
    def __init__(self):
        super(SACModel, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        self.critic1 = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.optimizer_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.optimizer_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)

    def train_dummy(self, episodes, steps_per_episode):
        total_steps = episodes * steps_per_episode
        for _ in range(total_steps):
            state = torch.rand(1, 6)
            action = self.actor(state)
            q_value1 = self.critic1(state)
            q_value2 = self.critic2(state)
            loss_actor = -q_value1.mean()
            loss_critic1 = ((q_value1 - 0.5) ** 2).mean()
            loss_critic2 = ((q_value2 - 0.5) ** 2).mean()

            # Backprop Actor
            self.optimizer_actor.zero_grad()
            loss_actor.backward(retain_graph=True)
            self.optimizer_actor.step()

            # Backprop Critic 1
            self.optimizer_critic1.zero_grad()
            loss_critic1.backward(retain_graph=True)
            self.optimizer_critic1.step()

            # Backprop Critic 2
            self.optimizer_critic2.zero_grad()
            loss_critic2.backward()
            self.optimizer_critic2.step()

# Utility to measure resources
def get_resource_usage():
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory().percent
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0].load * 100
            gpu_mem = gpus[0].memoryUtil * 100
            return cpu, ram, gpu, gpu_mem
    except:
        pass
    return cpu, ram, None, None

# Measure training time and resources
def measure_training(model, episodes, steps_per_episode, label):
    start_time = time.time()
    cpu_start, ram_start, gpu_start, gpu_mem_start = get_resource_usage()
    model.train_dummy(episodes, steps_per_episode)
    end_time = time.time()
    cpu_end, ram_end, gpu_end, gpu_mem_end = get_resource_usage()
    elapsed_time = end_time - start_time

    result = f"--- {label} Results ---\n" \
             f"Total Episodes: {episodes}\n" \
             f"Steps per Episode: {steps_per_episode}\n" \
             f"Training Time: {elapsed_time:.2f} seconds\n" \
             f"CPU Usage Start: {cpu_start}% End: {cpu_end}%\n" \
             f"RAM Usage Start: {ram_start}% End: {ram_end}%\n"
    if gpu_start is not None:
        result += f"GPU Usage Start: {gpu_start:.2f}% End: {gpu_end:.2f}%\n" \
                  f"GPU Memory Usage Start: {gpu_mem_start:.2f}% End: {gpu_mem_end:.2f}%\n"
    else:
        result += "GPU Usage: No GPU detected\n"
    result += "\n"
    print(result)
    return result

# Main testing
if __name__ == "__main__":
    episodes = 3000
    steps_per_episode = 10

    print("Starting PPO Training...")
    ppo_model = PPOModel()
    ppo_results = measure_training(ppo_model, episodes, steps_per_episode, "PPO")

    print("Starting SAC Training...")
    sac_model = SACModel()
    sac_results = measure_training(sac_model, episodes, steps_per_episode, "SAC")

    # Save results
    with open("drl_runtime_analysis_results.txt", "w") as f:
        f.write(ppo_results)
        f.write(sac_results)
    print("Results saved to drl_runtime_analysis_results.txt")
