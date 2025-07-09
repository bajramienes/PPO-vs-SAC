
# PPO vs SAC: Comparative Analysis for Energy Optimization with Country-Level Insights

This repository contains the full implementation and comparative analysis of two advanced Deep Reinforcement Learning (DRL) algorithms—Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC)—for national-scale energy optimization. The study uses real-world energy datasets to evaluate the performance of PPO and SAC in terms of stability, computational efficiency, and policy effectiveness. The project also explores regional comparisons (Balkan vs Nordic countries) for renewable integration and carbon intensity insights.

## 📂 Folder Structure

```
PPO_vs_SAC/
├── __pycache__/                  # Python cache files
├── countries_compare/            # Balkan vs Nordic comparison charts
├── NEW CHARTS/                   # Updated result charts
├── NEW CHARTS -1/                # Additional comparative charts
├── new-chart-1/                  # Boxplots and trend charts
├── results/                      # Algorithm metric charts (PPO vs SAC)
├── balkan_vs_nordic.py           # Regional comparison script
├── balkan_vs_nordic_specific.py  # Additional regional comparison logic
├── drl_metrics_comparison.pdf    # Comparative PDF report of DRL metrics
├── drl_runtime_analysis.py       # Runtime complexity analysis script
├── drl_runtime_analysis_results  # Saved results of runtime analysis
├── energy_drl_comparison.py      # DRL model comparison logic
├── evaluation.py                 # Evaluation metrics script
├── ppo_agent.py                  # PPO agent implementation
├── sac_agent.py                  # SAC agent implementation
├── ppo_vs_sac.py                 # PPO vs SAC main comparison script
├── ppo_vs_sac_metrics.py         # Metrics generation (variance, speed, scores)
├── requirements.txt              # Python dependencies
├── README.md                     # Project description file
└── World Energy Consumption.xlsx # Input dataset (2000–2023 country-level indicators)
```

## 🚀 Highlights

- **Agents:** PPO and SAC implemented using PyTorch and Stable Baselines 3.
- **Metrics:** Action variance, execution speed (episodes/sec), average action scores.
- **Regional Analysis:** Comparative insights between Balkan and Nordic countries.
- **Figures:** Publication-ready charts (PDF/SVG) for LaTeX integration.
- **Reproducibility:** Scripts aligned with Q1 journal standards.

## 🖥️ Running the Project

1. Clone the repository:
```bash
git clone https://github.com/bajramienes/PPO-vs-SAC.git
```

2. Navigate to the project folder:
```bash
cd PPO-vs-SAC
```

3. (Optional) Create a virtual environment:
```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Or on Linux/Mac
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run PPO vs SAC training:
```bash
python ppo_vs_sac.py
```

6. Generate metrics and regional charts:
```bash
python ppo_vs_sac_metrics.py
python balkan_vs_nordic.py
python new_chart_1.py
```

7. Measure computational complexity:
```bash
python drl_runtime_analysis.py
```

## 📊 Dataset

- **Source:** World Energy Consumption Dataset (Kaggle)  
- Covers indicators like greenhouse gas emissions, renewable/fossil share, electricity generation for 2000–2023.

## 📦 Requirements

- Python 3.8+
- Torch
- Stable Baselines 3
- Pandas, NumPy
- Matplotlib, Seaborn
- psutil, GPUtil (for runtime analysis)

## 📄 License

This repository is provided for academic purposes only. 

## 👨‍💻 Author

**Enes Bajrami**  
PhD Candidate 
Ss. Cyril and Methodius University - Faculty of Computer Science and Engineering (FCSE)
📧 enes.bajrami@students.finki.ukim.mk

