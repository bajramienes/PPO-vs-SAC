import os
import pandas as pd
import numpy as np
import torch
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from ppo_agent import PPOAgent
from sac_agent import SACAgent

# === Paths ===
DATA_PATH = r"C:/Users/Enes/Desktop/PPO vs SAC/World Energy Consumption.csv"
OUTPUT_DIR = r"C:/Users/Enes/Desktop/PPO vs SAC/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Country Groups ===
BALKAN = ["North Macedonia", "Kosovo", "Serbia", "Croatia", "Bosnia and Herzegovina"]
NORDIC = ["Denmark", "Norway", "Sweden", "Finland", "Iceland"]
REGIONS = BALKAN + NORDIC

# === Selected Features ===
FEATURES = [
    "electricity_generation", "greenhouse_gas_emissions",
    "renewables_share_energy", "fossil_share_energy",
    "solar_energy_per_capita", "oil_consumption"
]

# === Load Dataset ===
df = pd.read_csv(DATA_PATH)
df = df[df["country"].isin(REGIONS) & df["year"].between(2000, 2023)]
df["region"] = df["country"].apply(lambda x: "Balkan" if x in BALKAN else "Nordic")
df = df.fillna(0)

latest_year = df["year"].max()
latest_data = df[df["year"] == latest_year].copy()

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(latest_data[FEATURES])
X = pd.DataFrame(scaled_features, columns=FEATURES)
X["country"] = latest_data["country"].values
X["region"] = latest_data["region"].values

input_dim = len(FEATURES)
ppo_agent = PPOAgent(obs_dim=input_dim, act_dim=1)
sac_agent = SACAgent(obs_dim=input_dim, act_dim=1)

results = []
ppo_times, sac_times = [], []

# === Real-time testing loop with timing ===
for _, row in X.iterrows():
    country = row["country"]
    region = row["region"]
    obs = torch.tensor(row[FEATURES].values.astype(np.float32)).unsqueeze(0)

    # Timing PPO
    start_ppo = time.time()
    ppo_action, _ = ppo_agent.get_action(obs.numpy()[0])
    ppo_time = time.time() - start_ppo
    ppo_times.append(ppo_time)

    # Timing SAC
    start_sac = time.time()
    sac_action = sac_agent.actor(obs).detach().numpy()[0][0]
    sac_time = time.time() - start_sac
    sac_times.append(sac_time)

    results.append({
        "country": country,
        "region": region,
        "ppo_action": float(ppo_action[0]),
        "sac_action": float(sac_action)
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "ppo_sac_actions.csv"), index=False)

# === Speed calculation ===
ppo_speed = len(ppo_times) / sum(ppo_times)  # episodes per second
sac_speed = len(sac_times) / sum(sac_times)

# === Save speed results ===
speed_df = pd.DataFrame({
    "Algorithm": ["PPO", "SAC"],
    "Speed (Episodes per Second)": [ppo_speed, sac_speed]
})
speed_df.to_csv(os.path.join(OUTPUT_DIR, "ppo_sac_speed.csv"), index=False)

# === Chart for Action Comparison ===
region_avg = results_df.groupby("region")[["ppo_action", "sac_action"]].mean()
region_avg.plot(kind="bar", figsize=(8, 5), title="PPO vs SAC Action Comparison by Region")
plt.ylabel("Normalized Action (Efficiency Score)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ppo_vs_sac_comparison.svg"))
plt.close()

# === Speed Bar Chart ===
plt.figure(figsize=(6, 5))
plt.bar(["PPO", "SAC"], [ppo_speed, sac_speed], color=["blue", "orange"])
plt.title("Algorithm Speed: Episodes per Second")
plt.ylabel("Speed (Episodes/sec)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ppo_vs_sac_speed.svg"))
plt.close()

print("PPO vs SAC real-time test completed with speed logging.")
print("Results and charts saved to:", OUTPUT_DIR)
