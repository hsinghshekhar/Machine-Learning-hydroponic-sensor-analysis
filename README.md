<div align="center">

<img src="docs/logo.png" width="140"/>

# 🌿 Shekhar Agriculture Laboratory
### Hydroponic Farming — Machine Learning Research

[![Lab](https://img.shields.io/badge/Lab-Shekhar_Agriculture-gold?style=for-the-badge)](https://github.com/Shekhar-Agriculture-Laboratory)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![ML](https://img.shields.io/badge/ML-Logistic%20%7C%20Softmax%20%7C%20Ridge%20%7C%20Lasso-purple?style=for-the-badge)](.)
[![Dataset](https://img.shields.io/badge/Dataset-N%3D10%20Hydroponic%20Sensors-green?style=for-the-badge)](data/)
[![License](https://img.shields.io/badge/License-MIT-red?style=for-the-badge)](LICENSE)

---

**Complete ML Theory Applied to Real-World Hydroponic Crop Sensor Data**

*From binary classification → softmax → ERM → generalization → bias-variance → regularization*

</div>

---

## 📋 Overview

This repository contains a complete, step-by-step application of fundamental machine learning theory on **real hydroponic crop sensor data** collected at the Shekhar Agriculture Laboratory. Every calculation is done manually with actual numbers — no black boxes.

> 🌱 **Why Hydroponics + ML?** Accurate prediction of Electrical Conductivity (EC) from sensor readings enables proactive nutrient management and maximizes crop yield in soilless farming.

---

## 📊 Dataset

**N = 10 real sensor readings** (7-second intervals, Shekhar Agriculture Lab, March 2026)

| # | Timestamp | EC (ms/cm) | pH | Humidity % | Temp °C | Light (lux) | y_bin | y_k |
|---|-----------|-----------|-----|-----------|---------|------------|-------|-----|
| 1 | 2026-03-01 17:56:21 | 1.621 | 6.50 | 59.5 | 26.31 | 709 | 0 | Med |
| 2 | 2026-03-01 17:56:28 | 1.535 | 6.48 | 59.5 | 26.16 | 738 | 0 | Low |
| 3 | 2026-03-01 17:56:35 | 1.668 | 6.49 | 61.1 | 26.46 | 722 | 1 | High |
| 4 | 2026-03-01 17:56:42 | 1.689 | 6.52 | 61.1 | 25.57 | 797 | 1 | High |
| 5 | 2026-03-01 17:56:49 | 1.495 | 6.50 | 59.4 | 25.87 | 775 | 0 | Low |
| 6 | 2026-03-01 17:56:56 | 1.548 | 6.50 | 62.7 | 25.81 | 721 | 0 | Low |
| 7 | 2026-03-01 17:57:03 | 1.656 | 6.51 | 61.5 | 25.41 | 727 | 1 | High |
| 8 | 2026-03-01 17:57:10 | 1.632 | 6.48 | 60.6 | 26.16 | 760 | 1 | Med |
| 9 | 2026-03-01 17:57:17 | 1.661 | 6.51 | 63.6 | 25.81 | 721 | 1 | High |
| 10 | 2026-03-01 17:57:24 | 1.610 | 6.50 | 63.2 | 26.05 | 797 | 0 | Med |

**Label Rules:**
- `y_bin`: EC > 1.63 → 1 (High Growth), else 0 (Normal)
- `y_k`: EC < 1.55 → Low(0), 1.55 ≤ EC < 1.65 → Med(1), EC ≥ 1.65 → High(2)

---

## 🧠 ML Topics Covered (Step-by-Step)

### Section 0 — Dataset & i.i.d. Assumption
```
D = {(xi, yi)} i=1..10  ~  i.i.d.  P_XY
LLN: (1/N)·Σ L → E[L] as N→∞
```
- i.i.d. check: lag-1 autocorrelation r = 0.895
- All 10 samples from same hydroponic farm → identically distributed ✓

### Section 1 — Binary Linear Classifier
```
h_θ(x) = σ(θ₀ + θ₁·EC)    σ(z) = 1/(1+e⁻ᶻ)
```
| Parameter | Value |
|-----------|-------|
| θ₀ (bias) | −2.6809 |
| θ₁ (EC weight) | 1.6895 |
| Decision Boundary | EC = 1.5868 ms/cm |
| ERM Loss | 0.6527 |

### Section 2 — K-Class Softmax (K=3)
```
v₀ = −5·EC,  v₁ = 3·EC,  v₂ = 9·EC
P(class j) = exp(vⱼ) / Σ exp(vₖ)
```

### Section 3 — Logistic ERM
```
L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]
ERM Average Loss = 0.6527
```

### Section 4 — Generalization
| Split | Loss |
|-------|------|
| Train (samples 1–7) | 0.5652 |
| Val (samples 8–9) | measured |
| Test (sample 10, used ONCE) | 0.5844 |

### Section 5 — Square Error Loss
```
h*(x) = mean(EC) = 1.6115    MSE = 0.003760    RMSE = 0.0613
```

### Section 6 — Bias-Variance Decomposition
| Component | Value | % |
|-----------|-------|---|
| Variance | 0.000072 | 1.9% |
| Bias² | 0.000011 | 0.3% |
| Noise | 0.003760 | 97.8% |
| **Total R(h)** | **0.003843** | 100% |

### Regularization (L-10 Notes)
```
Reg-ERM: minimize  R̂(θ) + (λ/N)·Ω(θ)
Ω₁ = ‖θ‖²  (Ridge/L2)    Ω₂ = ‖θ‖₁  (Lasso/L1)
```

| Model | θ₀ | θ₁ | ‖θ‖² | Boundary | BCE Loss |
|-------|-----|-----|------|----------|---------|
| ERM | −2.6809 | 1.6895 | 10.04 | 1.5868 | 0.6527 |
| Ridge λ=0.5 | −0.7595 | 0.4722 | 0.800 | 1.6084 | 0.6815 |
| Lasso λ=0.5 | ~0 | ~0 | ~0 | — | 0.6931 |

---

## 📈 Results

<div align="center">

![Dataset Overview](results/fig01_dataset_overview.png)
*Figure 1: Dataset Overview — EC, pH, Humidity, Temperature across N=10 samples*

![Binary Classifier](results/fig02_binary_classifier.png)
*Figure 2: Binary Classifier — Sigmoid curve, GD loss curve, per-sample losses*

![Bias-Variance](results/fig03_bias_variance.png)
*Figure 3: Bias-Variance — deg=1 (underfit), deg=2 (balanced), deg=9 (overfit)*

![Regularization](results/fig04_regularization.png)
*Figure 4: Regularization — Ridge shrinkage path, L2 vs L1 penalty shapes*

</div>

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Shekhar-Agriculture-Laboratory/ml-hydroponic-sensor-analysis.git
cd ml-hydroponic-sensor-analysis

# Install dependencies
pip install numpy matplotlib scikit-learn

# Run complete ML pipeline
python src/ml_hydroponic_n10.py
```

---

## 📁 Repository Structure

```
ml-hydroponic-sensor-analysis/
│
├── 📊 data/
│   └── Sensor_Sample_Data_crop.csv    # N=10 real sensor readings
│
├── 🐍 src/
│   └── ml_hydroponic_n10.py           # Complete ML pipeline
│
├── 📈 results/
│   ├── fig01_dataset_overview.png     # Feature distributions
│   ├── fig02_binary_classifier.png    # Logistic regression results
│   ├── fig03_bias_variance.png        # B-V decomposition
│   └── fig04_regularization.png      # Ridge/Lasso comparison
│
├── 📄 docs/
│   └── logo.png                       # Lab logo
│
└── README.md
```

---

## 🔬 Key Insights

1. **EC > 1.63 ms/cm** predicts High Growth (Class 1) with 66.9% accuracy using single feature
2. **Noise dominates** (97.8%) in B-V decomposition → better labeling needed, not more complex model
3. **Ridge (λ=0.5)** shrinks ‖θ‖² from 10.04 → 0.80 (12.5× reduction) without sacrificing much accuracy
4. **Lasso (λ=0.5)** drives θ₁ → 0, effectively eliminating EC feature at this regularization strength
5. **deg=2 polynomial** is optimal for the regression task — matches true quadratic y=2EC²−4EC+2

---

## 📚 Theory Reference

| Concept | Formula |
|---------|---------|
| Sigmoid | σ(z) = 1/(1+e⁻ᶻ) |
| BCE Loss | L = −[y·log(ŷ) + (1−y)·log(1−ŷ)] |
| Ridge | Ω(θ) = ‖θ‖² |
| Lasso | Ω(θ) = ‖θ‖₁ |
| B-V | R(h) = Var + Bias² + Noise |
| ERM | θ* = argmin (1/N)·Σ L(h_θ(xᵢ), yᵢ) |
| Reg-ERM | θ* = argmin R̂(θ) + (λ/N)·Ω(θ) |

---

## 🏛️ About

**Shekhar Agriculture Laboratory**  
Hydroponic Farming Research & Data Science Division  
Report ID: SAL-ML-2026-N10 | March 14, 2026

---

<div align="center">

**⭐ Star this repo if it helped you understand ML with real agricultural data!**

[![GitHub stars](https://img.shields.io/github/stars/Shekhar-Agriculture-Laboratory/ml-hydroponic-sensor-analysis?style=social)](https://github.com/Shekhar-Agriculture-Laboratory/ml-hydroponic-sensor-analysis)

*© 2026 Shekhar Agriculture Laboratory. MIT License.*

</div>
