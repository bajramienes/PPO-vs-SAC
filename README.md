# PPO vs SAC for Country-Level Energy Optimization

[![Journal](https://img.shields.io/badge/Journal-IFAC%20Journal%20of%20Systems%20and%20Control-blue)](https://doi.org/10.1016/j.ifacsc.2025.100344)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ifacsc.2025.100344-green)](https://doi.org/10.1016/j.ifacsc.2025.100344)
[![Python](https://img.shields.io/badge/Python-3.11+-yellow)]()

Official implementation accompanying the paper:

> **A Comparative Analysis of PPO and SAC Algorithms for Energy Optimization with Country-Level Energy Consumption Insights**

Published in **IFAC Journal of Systems and Control (Elsevier), Volume 34, 2025**.

---

# Overview

This repository contains the experimental implementation used in the paper, where **Proximal Policy Optimization (PPO)** and **Soft Actor-Critic (SAC)** are evaluated for country-level energy optimization using real-world energy indicators.

Unlike studies relying solely on simulated environments, the proposed framework utilizes historical national energy data, including electricity generation, greenhouse gas emissions, renewable energy share, fossil fuel dependency, solar energy production, and oil consumption, enabling realistic evaluation across heterogeneous energy systems.

The repository includes:

- PPO implementation
- SAC implementation
- Rule-based baseline
- Multi-phase training and evaluation
- Regional analysis (Balkan vs Nordic countries)
- Publication-quality charts
- Experimental results
- Dataset preprocessing scripts

---

# Paper

Enes Bajrami, Andrea Kulakov, Eftim Zdravevski and Petre Lameski.

**A Comparative Analysis of PPO and SAC Algorithms for Energy Optimization with Country-Level Energy Consumption Insights**

IFAC Journal of Systems and Control, Volume 34, 2025.

DOI

https://doi.org/10.1016/j.ifacsc.2025.100344

---

# Repository Structure

```text
.
├── charts/                         Publication-quality figures used in the paper
├── data/                           World Energy Consumption dataset
├── results/                        Experimental outputs and evaluation results
├── src/                            Source code (PPO, SAC, evaluation, analysis)
├── README.md                       Repository documentation
├── requirements.txt                Python dependencies
└── drl_runtime_analysis_results.txt Runtime analysis results
```

---

# Evaluated Algorithms

The following methods are evaluated in this study:

- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)
- Rule-based baseline

---

# Experimental Environment

- Python
- PyTorch
- Stable-Baselines3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Experiments were performed using real-world country-level energy data covering multiple sustainability indicators and evaluated across several training phases ranging from 300 to 3000 episodes.

---

# Dataset

The framework uses the **World Energy Consumption** dataset containing national energy indicators including:

- Electricity generation
- Greenhouse gas emissions
- Renewable energy share
- Fossil fuel dependency
- Solar energy production
- Oil consumption

The paper also includes a regional comparison between Balkan and Nordic countries using these indicators.

---

# Evaluation

The paper evaluates PPO and SAC using:

- Cumulative Reward
- Action Variance
- Execution Speed
- Average Action Score
- Regional Performance Comparison
- Policy Stability

---

# Citation

If you use this repository in your research, please cite:

```bibtex
@article{BAJRAMI2025100344,
  title   = {A comparative analysis of PPO and SAC algorithms for energy optimization with country-level energy consumption insights},
  journal = {IFAC Journal of Systems and Control},
  volume  = {34},
  pages   = {100344},
  year    = {2025},
  issn    = {2468-6018},
  doi     = {10.1016/j.ifacsc.2025.100344},
  url     = {https://www.sciencedirect.com/science/article/pii/S2468601825000501},
  author  = {Enes Bajrami and Andrea Kulakov and Eftim Zdravevski and Petre Lameski}
}
```

---

# License

This repository is released for academic and research purposes.

Please cite the associated publication when using this code or any derived results.

---

# Contact

**Enes Bajrami**

Faculty of Computer Science and Engineering (FINKI)

Ss. Cyril and Methodius University in Skopje

Email

enes.bajrami@students.finki.ukim.mk
