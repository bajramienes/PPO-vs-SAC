import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.preprocessing import MinMaxScaler

# === Setup ===
dataset_path = r"C:/Users/Enes/Desktop/PPO vs SAC/World Energy Consumption.csv"
output_dir = r"C:/Users/Enes/Desktop/PPO vs SAC/countries_compare"
os.makedirs(output_dir, exist_ok=True)

# === Country groups ===
balkan = ["North Macedonia", "Kosovo", "Serbia", "Croatia", "Bosnia and Herzegovina"]
nordic = ["Denmark", "Norway", "Sweden", "Finland", "Iceland"]
df = pd.read_csv(dataset_path)

# === Filter and preprocess ===
df = df[df["country"].isin(balkan + nordic)]
df = df[df["year"].between(2000, 2023)]
df["region"] = df["country"].apply(lambda x: "Balkan" if x in balkan else "Nordic")
df = df.fillna(0)

# === Features to compare ===
features = [
    "electricity_generation", "greenhouse_gas_emissions",
    "renewables_share_energy", "fossil_share_energy",
    "solar_electricity", "oil_consumption"
]

# === Time series plots per feature ===
for feature in features:
    plt.figure(figsize=(12, 6))
    for region in ["Balkan", "Nordic"]:
        regional = df[df["region"] == region].groupby("year")[feature].mean().reset_index()
        plt.plot(regional["year"], regional[feature], label=region)
    plt.title(f"{feature.replace('_', ' ').title()} (2000–2023)")
    plt.xlabel("Year")
    plt.ylabel(feature.replace('_', ' ').title())
    plt.legend()
    plt.xticks(ticks=list(range(2000, 2024)), rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{feature}_balkan_vs_nordic.svg"))
    plt.close()

# === Radar chart for the latest year ===
latest_year = df["year"].max()
latest_data = df[df["year"] == latest_year].copy()
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(latest_data[features])
latest_scaled = pd.DataFrame(scaled_features, columns=features)
latest_scaled["region"] = latest_data["region"].values

radar_data = latest_scaled.groupby("region").mean().reset_index()

labels = features
angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
for idx, row in radar_data.iterrows():
    values = row[features].tolist()
    values += values[:1]
    ax.plot(angles, values, label=row["region"])
    ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=9)
plt.title("Balkan vs Nordic Energy Profile Radar (Latest Year)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "balkan_vs_nordic_radar.svg"))
plt.close()

print("Balkan vs Nordic charts generated successfully in:", output_dir)
