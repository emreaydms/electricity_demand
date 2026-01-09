# Short-Term Electricity Demand Forecasting with Hybrid Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning pipeline for short-term electricity load forecasting (STLF) using real-world European power system data. This project systematically progresses from classical baselines to modern hybrid architectures, culminating in a **Genetic Algorithm-optimized CNN-TCN-GRU-Attention model**.

The work was developed as a graduate-level deep learning project and structured to be reproducible, extensible, and research-oriented.

---

## Project Highlights

- **Real-world dataset**: Hourly electricity demand for Hungary (ENTSO-E), pre-COVID period
- **Multiple model families**: MLP, CNN, TCN, LSTM, Attention mechanisms, Hybrid architectures
- **Genetic Algorithm optimization** for automated deep architecture search
- **Temporal modeling** at multiple scales (hourly, daily, weekly patterns)
- **Extensive visual analysis** with error diagnostics and failure mode investigation
- **Modular pipeline**: Data → Features → Models → Evaluation (production-style code)

---

## Problem Definition

### Task
Given the past **24 hours** of multivariate time-series data, predict the **next-hour electricity demand**:

```
X_{t-24:t} → ŷ_{t+1}
```

### Formulation
- **Type**: Supervised regression problem
- **Input**: 24-hour sliding window with multivariate features
- **Output**: Single-step ahead demand forecast
- **Temporal granularity**: Hourly

### Evaluation Metrics
1. **MAPE** (Mean Absolute Percentage Error) - Primary metric
2. **MAE** (Mean Absolute Error)
3. **RMSE** (Root Mean Squared Error)

---

## Repository Structure

```
electricity-forecasting/
│
├── notebooks/
│   ├── data.ipynb                        # Data merging & cleaning
│   ├── feature_engineering.ipynb         # Lag, calendar, HDD/CDD features
│   │
│   ├── model_training_dense.ipynb        # MLP baseline
│   ├── model_training_cnn1d.ipynb        # CNN-1D baseline
│   ├── model_training_tcn.ipynb          # Temporal Convolutional Network
│   ├── model_training_TimesNet_TCN.ipynb # TimesNet + TCN hybrid
│   ├── model_training_attention.ipynb    # CNN-LSTM-Attention
│   ├── model_training_GA.ipynb           # GA-optimized hybrid model
│   │
│   ├── 03_TimesNet_TCN.ipynb            # Alternative TimesNet implementation
│   ├── 04_attention.ipynb               # Attention mechanism experiments
│   ├── 05_GA.ipynb                      # Genetic Algorithm optimization
│
├── src/
│   ├── entsoe_data_fetcher.py           # ENTSO-E load data retrieval
│   ├── weather_data_fetcher.py          # Weather API integration
│   ├── feature_engineering.py           # Feature pipeline
│   ├── calendar_generator.py            # Holidays & calendar logic
│   ├── models/
│
├── report.pdf                           # Final IEEE-style technical report
```

---

## Model Architectures

### 1️Baseline Models

#### MLP (Dense Network)
```python
Input Features → Dense(256) → ReLU → Dropout
                → Dense(128) → ReLU → Dropout
                → Dense(64)  → ReLU
                → Dense(1)   → Output
```

**Characteristics:**
- Tabular model using engineered lag features
- Fast training, interpretable
- Limited temporal awareness
- **MAPE: 2.71%**

#### CNN-1D (Convolutional)
```python
Input Sequence → Conv1D(32, k=3) → ReLU → BatchNorm
               → Conv1D(64, k=3) → ReLU → BatchNorm
               → GlobalAvgPool
               → Dense(32) → Output
```

**Characteristics:**
- Local temporal pattern extraction
- Translation invariant
- Struggles with long-range dependencies
- **MAPE: 2.15%**

---

### 2️Literature-Inspired Architectures

#### TimesNet + TCN

```python
Input → TimesNet Block (2D periodic decomposition)
      → TCN Residual Blocks (dilation=1,2,4,8)
      → Global Context Pooling
      → Dense → Output
```

**Key Features:**
- TimesNet-style 2D convolution for periodic patterns
- TCN residual blocks for temporal dynamics
- Exponentially growing receptive field via dilated convolutions
- Strong improvement in weekly pattern tracking

**Performance:**
- **MAPE: 1.91%**
- Excellent at capturing daily/weekly cycles
- Occasional smoothing at demand peaks

---

#### CNN-LSTM-Attention

```python
Input → CNN Feature Extractor (64 filters)
      → LSTM (hidden=128, layers=2)
      → Multi-Head Attention (heads=4)
      → Weighted Temporal Fusion
      → Dense → Output
```

**Key Features:**
- CNN for low-level feature extraction
- LSTM for sequential dependency modeling
- Attention for temporal importance weighting
- Better local detail preservation

**Performance:**
- **MAPE: 1.89%**
- Captures fine-grained temporal patterns
- Unstable at extreme values (peaks/valleys)

---

### 3️Proposed Hybrid Model (Main Contribution)

#### Architecture: CNN → TCN → GRU → Multi-Head Attention → Fusion

```python
# Stage 1: Local Feature Extraction
x = CNN_Block(input)                    # Short-term patterns
  → Conv1D(32, k=3) → ReLU → BatchNorm
  → Conv1D(64, k=3) → ReLU → BatchNorm

# Stage 2: Mid-Range Temporal Modeling
x = TCN_Residual_Stack(x)              # Medium-term dependencies
  → [Dilated Conv (d=1,2,4,8) + Residual] × 4 blocks

# Stage 3: Long-Range Sequential Modeling
x, hidden = GRU(x, hidden_dim=128)     # Long-term context

# Stage 4: Temporal Attention
attn_weights = MultiHeadAttention(x, heads=8)
context = weighted_sum(x, attn_weights)

# Stage 5: Adaptive Feature Fusion
gate = σ(W_fusion · [x, context])
output = gate ⊙ x + (1-gate) ⊙ context
  → Dense(1) → Prediction
```

#### Key Design Principles

1. **Hierarchical Temporal Modeling**
   - CNN: Captures hourly variations and immediate patterns
   - TCN: Models daily cycles with exponential receptive fields
   - GRU: Encodes weekly trends and long-term dependencies

2. **Attention Mechanism**
   - Multi-head design captures different temporal aspects
   - Learns which time steps are most predictive
   - Provides interpretability via attention weights

3. **Adaptive Fusion**
   - Gated mechanism balances CNN/TCN features vs attention-weighted context
   - Model learns optimal combination per time step
   - Prevents over-reliance on single pathway

4. **Residual Connections**
   - Facilitates gradient flow in deep architecture
   - Enables stable training of 10+ layer model
   - Preserves information across temporal scales

---

## Genetic Algorithm Optimization

### Motivation
Instead of manual hyperparameter tuning, the final architecture is **automatically optimized** using a Genetic Algorithm, evolving:

- CNN channel sizes: [16, 32, 64, 128]
- TCN channel sizes: [32, 64, 128, 256]
- GRU hidden dimension: [64, 128, 256]
- Attention head count: [2, 4, 8, 16]
- Kernel sizes: [3, 5, 7]
- Dropout rates: [0.0, 0.1, 0.2, 0.3]
- Layer normalization: [True, False]
- Number of TCN blocks: [2, 3, 4, 5]

### Why Genetic Algorithm?

**Handles discrete choices**: Architecture components (layer types, counts)  
**Non-differentiable optimization**: Can't use gradient descent for discrete decisions  
**Exploration**: Population-based search avoids local optima  
**Flexibility**: More suitable than PSO for architecture search  
**Produces diverse solutions**: Multiple high-performing configurations

### GA Configuration

```python
Population size: 20 individuals
Generations: 25
Selection: Tournament (k=3)
Crossover: Two-point crossover (p=0.8)
Mutation: Random gene flip (p=0.1)
Fitness: Validation MAPE (minimize)
Elitism: Top 2 individuals preserved
```

### Optimization Results

- **Runtime**: ~50 minutes (on NVIDIA RTX 3080)
- **Best validation MAPE**: 1.88% (Generation 18)
- **Final test MAPE**: 1.80%

#### Best Architecture Found
```
CNN: [32, 64] channels, kernel=5
TCN: [64, 128, 128] channels, dilation=[1,2,4], kernel=3
GRU: hidden_dim=128
Attention: 8 heads
Dropout: 0.2
Layer Norm: True
```

---

## Results Summary

### Quantitative Performance

| Model | MAPE (%) ↓ | MAE | RMSE | Training Time |
|-------|-----------|-----|------|---------------|
| **MLP** | 2.71 | 134.2 | 189.5 | 5 min |
| **CNN-1D** | 2.15 | 106.8 | 152.3 | 8 min |
| **TimesNet-TCN** | 1.91 | 94.7 | 138.2 | 15 min |
| **CNN-LSTM-Attn** | 1.89 | 93.5 | 136.8 | 20 min |
| **GA Hybrid** | **1.80** | **89.1** | **130.4** | 50 min (GA) + 18 min (train) |

**Key Observations:**
- Hybrid temporal models significantly outperform baselines
- GA-optimized model achieves best overall accuracy
- 33.6% MAPE reduction compared to MLP baseline
- 16.3% improvement over standard CNN-1D

---

### Qualitative Analysis

#### Strengths
1. **Excellent trend following**: Captures overall demand patterns with high fidelity
2. **Strong periodicity modeling**: Daily and weekly cycles accurately represented
3. **Smooth predictions**: Reduced noise compared to baselines
4. **Robust to weather variations**: Incorporates HDD/CDD features effectively

#### Limitations
1. **Peak underprediction**: Conservative at extreme high demand (heatwaves, cold snaps)
2. **Valley overshooting**: Tends to underestimate depth of low-demand periods
3. **Transition lag**: Slight delay in responding to abrupt demand shifts (holidays)
4. **Extreme event sensitivity**: Struggles with unprecedented demand patterns

---

## Key Findings

### 1. Temporal Inductive Bias is Critical
- Pure MLP treats time as tabular data → poor performance
- TCN and RNN architectures explicitly model temporal structure
- **Result**: 20-30% error reduction with temporal models

### 2. Attention Improves Local Detail
- Attention mechanism focuses on relevant historical periods
- Better prediction of short-term fluctuations
- **Trade-off**: May smooth extreme peaks (conservative predictions)

### 3. Hybrid Architectures Provide Best Balance
- Single-architecture models have blind spots
- CNN captures local, TCN captures periodic, GRU captures long-term
- **Synergy**: Combined model covers all temporal scales

### 4. Extreme Peak/Valley Prediction Remains Challenging
- All models underpredict demand spikes (> 2σ above mean)
- Caused by:
  - Distribution shift (rare events in training)
  - Conservative loss functions (MSE penalizes large errors)
  - Model capacity limitations
- **Future direction**: Asymmetric loss, quantile regression

---

## Data Description

### Primary Dataset: ENTSO-E

**Source**: European Network of Transmission System Operators for Electricity  
**Region**: Hungary (HU)  
**Period**: January 2017 - December 2019 (pre-COVID)  
**Resolution**: Hourly  
**Size**: 26,280 samples

**Features:**
- `datetime`: Timestamp (hourly)
- `demand_MW`: Total electricity demand in megawatts
- `temperature_C`: Ambient temperature (°C)
- `wind_speed_mps`: Wind speed (m/s)
- `solar_radiation_Wm2`: Solar radiation (W/m²)

### Engineered Features

#### 1. Temporal Lags (24 features)
```python
# Previous 24 hours of demand
demand_t-1, demand_t-2, ..., demand_t-24
```

#### 2. Calendar Features (7 features)
```python
- hour_of_day (0-23)
- day_of_week (0-6)
- day_of_month (1-31)
- month (1-12)
- is_weekend (binary)
- is_holiday (binary, Hungarian holidays)
- season (0-3: Winter, Spring, Summer, Fall)
```

#### 3. Heating/Cooling Degree Days
```python
# HDD: Heating demand proxy
HDD = max(0, 18°C - temperature)

# CDD: Cooling demand proxy
CDD = max(0, temperature - 24°C)
```

#### 4. Cyclical Encodings
```python
# Sine/cosine transformations for periodicity
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
day_sin = sin(2π * day / 7)
day_cos = cos(2π * day / 7)
```

### Data Split

```python
Train: 70% (Jan 2017 - Sep 2018)  →  18,396 samples
Val:   15% (Oct 2018 - Mar 2019)  →   3,942 samples
Test:  15% (Apr 2019 - Dec 2019)  →   3,942 samples
```

**Important**: Time-series split (no shuffling) to prevent data leakage.

---

## Experimental Setup

### Training Configuration

```python
# Optimization
optimizer = Adam(lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(patience=5, factor=0.5)
loss_function = MSELoss()

# Training
batch_size = 64
epochs = 100
early_stopping_patience = 15

# Hardware
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mixed_precision = True  # FP16 training for speedup
```

### Data Augmentation

```python
# Gaussian noise injection (training only)
noise_std = 0.01 * demand_std

# Random temporal shifting (±2 hours)
shift_range = [-2, -1, 0, 1, 2]

# Mixup augmentation (α=0.2)
λ ~ Beta(0.2, 0.2)
x_mix = λ * x_i + (1-λ) * x_j
```

---

## Future Improvements

### 1. Peak-Aware Loss Functions
```python
# Asymmetric loss to penalize underprediction
def asymmetric_mse(pred, target, alpha=2.0):
    error = target - pred
    loss = torch.where(
        error > 0,
        alpha * error**2,  # Higher penalty for underprediction
        error**2
    )
    return loss.mean()
```

### 2. Regime-Based Modeling
- Separate models for: normal, peak, valley regimes
- Gating network to select appropriate model
- Ensemble predictions for robustness

### 3. Multi-Task Learning
```python
# Jointly predict: demand, trend, volatility
outputs = model(x)
demand_pred = outputs[0]
trend_pred = outputs[1]
volatility_pred = outputs[2]
```

### 4. Probabilistic Forecasting
- Quantile regression for uncertainty bounds
- Monte Carlo Dropout for confidence intervals
- Gaussian Process posterior over predictions

### 5. Multi-Country Generalization
- Transfer learning from Hungary → other EU countries
- Domain adaptation techniques
- Shared encoder, country-specific decoders

### 6. Real-Time Deployment
- Model quantization (INT8) for edge devices
- ONNX export for cross-platform inference
- REST API with FastAPI
- Continuous learning pipeline

---

## Report & Citation

### Technical Report

A comprehensive **IEEE-style technical report** is included as [`report.pdf`](report.pdf), covering:

1. **Introduction**: Problem formulation and motivation
2. **Literature Review**: Survey of STLF methods
3. **Dataset**: ENTSO-E data description and EDA
4. **Methodology**: Architecture designs and training procedures
5. **Experiments**: Ablation studies and hyperparameter analysis
6. **Results**: Quantitative metrics and qualitative analysis
7. **Discussion**: Failure modes and future directions
8. **Conclusion**: Summary and contributions

---

## 👥 Authors

### Emre Aydoğmuş
- 🎓 AI & Data Engineering, Istanbul Technical University

### Baturalp Taha Yılmaz
- 🎓 AI & Data Engineering, Istanbul Technical University


---

## References

1. **TimesNet**: Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis", ICLR 2023
2. **TCN**: Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", arXiv 2018
3. **Attention**: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
4. **ENTSO-E Platform**: [https://transparency.entsoe.eu/](https://transparency.entsoe.eu/)
5. **Short-Term Load Forecasting**: Hong et al., "Energy Forecasting: Past, Present, and Future", Foresight 2016

---

<div align="center">

**⚡ Powering the future with intelligent forecasting ⚡**

Made with ❤️ by Emre Aydoğmuş

[⬆ Back to Top](#-short-term-electricity-demand-forecasting-with-hybrid-deep-learning)

</div>
