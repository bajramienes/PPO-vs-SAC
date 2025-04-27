import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.preprocessing import MinMaxScaler

# === Paths ===
DATA_PATH = r"C:/Users/Enes/Desktop/PPO vs SAC/World Energy Consumption.csv"
OUTPUT_DIR = r"C:/Users/Enes/Desktop/PPO vs SAC/new-chart-1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Country Groups ===
BALKAN = ["North Macedonia", "Kosovo", "Serbia", "Croatia", "Bosnia and Herzegovina"]
NORDIC = ["Denmark", "Norway", "Sweden", "Finland", "Iceland"]
ALL_COUNTRIES = BALKAN + NORDIC

# === Selected Features ===
FEATURES = [
    "electricity_generation", "greenhouse_gas_emissions",
    "renewables_share_energy", "fossil_share_energy",
    "solar_energy_per_capita", "oil_consumption"
]

# === Load Dataset ===
df = pd.read_csv(DATA_PATH)
df = df[df["country"].isin(ALL_COUNTRIES) & df["year"].between(2000, 2023)]
df["region"] = df["country"].apply(lambda x: "Balkan" if x in BALKAN else "Nordic")
df[FEATURES] = df[FEATURES].fillna(0)

# === Chart 1: Average Energy Indicators per Country (Barplot) ===
avg_df = df.groupby("country")[FEATURES].mean().reset_index()
melted_avg = avg_df.melt(id_vars="country", var_name="feature", value_name="value")

plt.figure(figsize=(14, 7))
sns.barplot(data=melted_avg, x="country", y="value", hue="feature")
plt.title("Average Energy Indicators per Country (2000–2023)")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "average_energy_indicators_per_country.svg"))
plt.close()

# === Chart 2: Lineplot of Electricity Generation over Time ===
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x="year", y="electricity_generation", hue="country", marker="o")
plt.title("Electricity Generation Over Time (2000–2023)")
plt.ylabel("TWh")
plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(ticks=list(range(2000, 2024)), rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "electricity_generation_time_series.svg"))
plt.close()

# === Chart 3: Boxplot for Greenhouse Gas Emissions ===
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="country", y="greenhouse_gas_emissions")
plt.title("Greenhouse Gas Emissions Distribution (2000–2023)")
plt.ylabel("Million Tonnes CO2")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "ghg_emissions_boxplot.svg"))
plt.close()

# === Chart 4: Radar Chart (Region-Level Average Profile) ===
region_avg = df.groupby("region")[FEATURES].mean()
scaler = MinMaxScaler()
normalized = scaler.fit_transform(region_avg)
radar_df = pd.DataFrame(normalized, index=region_avg.index, columns=region_avg.columns)

labels = radar_df.columns.tolist()
angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
angles += angles[:1]  # loop

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
for idx, row in radar_df.iterrows():
    values = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, values, label=idx)
    ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
plt.title("Radar Chart: Balkan vs Nordic Energy Profiles")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "region_radar_profile.svg"))
plt.close()

print("Balkan vs Nordic charts successfully generated in:", OUTPUT_DIR)
