import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
RESULTS_DIR = r"C:/Users/Enes/Desktop/PPO vs SAC/results"
ACTIONS_FILE = os.path.join(RESULTS_DIR, "ppo_sac_actions.csv")
SPEED_FILE = os.path.join(RESULTS_DIR, "ppo_sac_speed.csv")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "metrics_charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load data ===
actions_df = pd.read_csv(ACTIONS_FILE)
speed_df = pd.read_csv(SPEED_FILE)

# === Prepare data ===
# Reshape actions for easier plotting
melted_actions = actions_df.melt(id_vars=["country", "region"], 
                                 value_vars=["ppo_action", "sac_action"], 
                                 var_name="Algorithm", 
                                 value_name="Action")

# === 1. Average Action Score per Algorithm ===
avg_actions = melted_actions.groupby("Algorithm")["Action"].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=avg_actions, x="Algorithm", y="Action")
plt.title("Average Action Score (PPO vs SAC)")
plt.ylabel("Mean Action (Efficiency Score)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "average_action_score.svg"))
plt.close()

# === 2. Speed (Episodes per Second) ===
plt.figure(figsize=(8, 5))
sns.barplot(data=speed_df, x="Algorithm", y="Speed (Episodes per Second)")
plt.title("Algorithm Speed: Episodes per Second")
plt.ylabel("Speed (episodes/sec)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "algorithm_speed.svg"))
plt.close()

# === 3. Action Variance (Reward Stability) ===
variance_df = melted_actions.groupby("Algorithm")["Action"].var().reset_index(name="Action Variance")

plt.figure(figsize=(8, 5))
sns.barplot(data=variance_df, x="Algorithm", y="Action Variance")
plt.title("Action Variance (Stability Indicator)")
plt.ylabel("Variance of Action Scores")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "action_variance.svg"))
plt.close()

# === 4. Radar Chart: Combined Metrics Summary ===
from math import pi
radar_data = pd.DataFrame({
    "Metric": ["Average Action", "Action Stability (1/Variance)", "Speed (Episodes/sec)"],
    "PPO": [
        avg_actions[avg_actions["Algorithm"] == "ppo_action"]["Action"].values[0],
        1 / variance_df[variance_df["Algorithm"] == "ppo_action"]["Action Variance"].values[0],
        speed_df[speed_df["Algorithm"] == "PPO"]["Speed (Episodes per Second)"].values[0]
    ],
    "SAC": [
        avg_actions[avg_actions["Algorithm"] == "sac_action"]["Action"].values[0],
        1 / variance_df[variance_df["Algorithm"] == "sac_action"]["Action Variance"].values[0],
        speed_df[speed_df["Algorithm"] == "SAC"]["Speed (Episodes per Second)"].values[0]
    ]
})

# Prepare data for radar plot
labels = radar_data["Metric"].tolist()
num_vars = len(labels)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)] + [0]

ppo_values = radar_data["PPO"].tolist() + [radar_data["PPO"].tolist()[0]]
sac_values = radar_data["SAC"].tolist() + [radar_data["SAC"].tolist()[0]]

plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)
ax.plot(angles, ppo_values, linewidth=2, label="PPO")
ax.fill(angles, ppo_values, alpha=0.25)

ax.plot(angles, sac_values, linewidth=2, label="SAC")
ax.fill(angles, sac_values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=9)
plt.title("PPO vs SAC Metrics Summary (Radar Chart)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ppo_vs_sac_radar_summary.svg"))
plt.close()

print("PPO vs SAC metrics comparison completed successfully.")
print("Charts saved in:", OUTPUT_DIR)
