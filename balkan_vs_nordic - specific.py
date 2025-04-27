import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.preprocessing import MinMaxScaler

# === Setup ===
dataset_path = r"C:/Users/Enes/Desktop/PPO vs SAC/World Energy Consumption.csv"
output_dir = r"C:/Users/Enes/Desktop/PPO vs SAC/new_charts"
os.makedirs(output_dir, exist_ok=True)

# === Define Regions ===
balkan = ["North Macedonia", "Kosovo", "Serbia", "Croatia", "Bosnia and Herzegovina"]
nordic = ["Denmark", "Norway", "Sweden", "Finland", "Iceland"]
regions = balkan + nordic

# === Load Dataset ===
df = pd.read_csv(dataset_path)
df = df[df["country"].isin(regions) & df["year"].between(2000, 2023)]
df["region"] = df["country"].apply(lambda x: "Balkan" if x in balkan else "Nordic")
df = df.fillna(0)

# === Selected Features ===
features = [
    "electricity_generation", "greenhouse_gas_emissions",
    "renewables_share_energy", "fossil_share_energy",
    "solar_energy_per_capita", "oil_consumption"
]

# === 1. Grouped Barplot: Average Features per Country ===
avg_features = df.groupby("country")[features].mean().reset_index()
avg_features = pd.melt(avg_features, id_vars="country", var_name="Feature", value_name="Value")

plt.figure(figsize=(14, 7))
sns.barplot(data=avg_features, x="country", y="Value", hue="Feature")
plt.title("Average Energy Indicators per Country (2000–2023)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "grouped_barplot_energy_indicators.svg"))
plt.close()

# === 2. Radar Chart: Region Profiles ===
scaler = MinMaxScaler()
region_avg = df.groupby("region")[features].mean()
region_scaled = pd.DataFrame(scaler.fit_transform(region_avg), index=region_avg.index, columns=features)

angles = [n / float(len(features)) * 2 * pi for n in range(len(features))] + [0]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
for region in region_scaled.index:
    values = region_scaled.loc[region].tolist() + [region_scaled.loc[region].tolist()[0]]
    ax.plot(angles, values, label=region)
    ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=9)
plt.title("Radar Chart: Region Profiles (Average Energy Indicators)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "region_radar_profiles.svg"))
plt.close()

# === 3. Stacked Area Chart: Fossil vs Renewables Over Time ===
df_yearly = df.groupby(["year", "region"])[["fossil_share_energy", "renewables_share_energy"]].mean().reset_index()
pivoted = df_yearly.pivot(index="year", columns="region")

plt.figure(figsize=(12, 7))
plt.stackplot(
    df_yearly["year"].unique(),
    pivoted["fossil_share_energy"]["Balkan"],
    pivoted["fossil_share_energy"]["Nordic"],
    labels=["Balkan Fossil Share", "Nordic Fossil Share"]
)
plt.title("Fossil Energy Share Over Time (2000–2023)")
plt.legend(loc="upper right")
plt.xlabel("Year")
plt.ylabel("Fossil Share (%)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fossil_energy_share_stacked_area.svg"))
plt.close()

# === 4. Heatmap of Correlations: Balkan vs Nordic ===
for region_name in ["Balkan", "Nordic"]:
    subset = df[df["region"] == region_name][features]
    corr = subset.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Heatmap - {region_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{region_name.lower()}_correlation_heatmap.svg"))
    plt.close()

# === 5. Violin Plot: Fossil Share Distribution ===
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="region", y="fossil_share_energy")
plt.title("Fossil Energy Share Distribution (2000–2023)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fossil_share_violinplot.svg"))
plt.close()

# === 6. Line Plot: Solar Energy Per Capita Trend ===
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="year", y="solar_energy_per_capita", hue="country", style="region")
plt.title("Solar Energy Per Capita (2000–2023)")
plt.xlabel("Year")
plt.ylabel("MWh per capita")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "solar_energy_trend.svg"))
plt.close()

print("Balkan vs Nordic specific comparison charts generated successfully.")
print("Charts saved in:", output_dir)
