# 🛡️ Lightweight Explainable IDS for IoT Devices

> XGBoost-based Intrusion Detection System with SHAP explainability and real-time SOC triage dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-purple)

---

## 📌 Overview

This project implements a lightweight, explainable Intrusion Detection System (IDS) designed for IoT network traffic. It uses **XGBoost** for binary classification of network flows as Normal or Attack, **SHAP** for model explainability, and a **Streamlit** dashboard for real-time SOC analyst triage.

The key novel contribution is a **SHAP-Weighted Alert Priority Scoring (SWAPS)** system that ranks alerts not just by model confidence, but by a composite score incorporating:
- Model prediction confidence
- SHAP magnitude (how strongly features push toward attack)
- Domain-knowledge-informed feature criticality weights

---

## 🗂️ Project Structure

```
iot-ids-xgboost/
├── app.py                      # Main Streamlit dashboard
├── realtime_input.py           # Real-time / manual flow input module
├── alert_priority.py           # SHAP-weighted alert priority scoring
├── models/
│   ├── xgboost_ids.pkl         # Trained XGBoost model
│   ├── scaler.pkl              # StandardScaler
│   └── threshold.pkl           # Optimal classification threshold (0.7)
├── notebooks/
│   ├── 01_eda.ipynb            # EDA, preprocessing, model training
│   └── 02_multiclass.ipynb     # Multiclass attack classification
├── data/
│   ├── UNSW_NB15_training-set.parquet
│   └── UNSW_NB15_testing-set.parquet
└── outputs/
    ├── confusion_matrix.png
    └── shap_summary.png
```

---

## 📊 Dataset

**UNSW-NB15** — A comprehensive network intrusion dataset containing real modern attack behaviours and synthetic normal traffic.

| Split | Rows | Attack % |
|-------|------|----------|
| Training | 175,341 | 68.1% |
| Testing  | 82,332  | 55.1% |

Features used: 34 network flow features including duration, packet counts, byte counts, TTL values, jitter, window sizes, and connection-level statistics.

---

## 🤖 Model Performance

### Binary Classification (XGBoost)

| Metric | Value |
|--------|-------|
| Accuracy | 90% |
| Precision (Attack) | 91.5% |
| Recall (Attack) | 92.8% |
| F1-Score | 92.2% |
| False Positive Rate | 10.5% |

> Threshold tuned to **0.7** (from default 0.5) to reduce false positive rate from 30.4% to 10.5% — critical for SOC analyst efficiency.

### Threshold Analysis

| Threshold | Precision | Recall | F1 | FP Rate |
|-----------|-----------|--------|----|---------|
| 0.3 | 0.750 | 0.997 | 0.856 | 40.7% |
| 0.4 | 0.765 | 0.992 | 0.864 | 37.3% |
| 0.5 | 0.798 | 0.981 | 0.880 | 30.4% |
| 0.6 | 0.854 | 0.958 | 0.903 | 20.0% |
| **0.7** | **0.915** | **0.928** | **0.922** | **10.5%** |

---

## 🚨 SHAP-Weighted Alert Priority Scoring (SWAPS)

**Priority Score = α × Confidence + β × SHAP Magnitude + γ × Critical Feature Boost**

| Component | Weight | Description |
|-----------|--------|-------------|
| α — Model Confidence | 40% | predict_proba score |
| β — SHAP Magnitude | 40% | Normalised sum of absolute SHAP values |
| γ — Criticality Boost | 20% | Domain-knowledge feature weights |

**Severity Levels:**

| Score | Severity | Action |
|-------|----------|--------|
| ≥ 80 | 🔴 CRITICAL | Auto-escalate to L2/L3 |
| 60–79 | 🟠 HIGH | Immediate investigation |
| 40–59 | 🟡 MEDIUM | Queue for analysis |
| < 40 | 🟢 LOW | Log and monitor |

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard | Dataset overview — metrics, charts, attack breakdown |
| 📝 Manual Analysis | Enter flow features manually, get instant triage card |
| ⚡ Live Stream | Simulated real-time flow analysis with priority scoring |
| 📈 Alert Queue | All analysed flows sorted by SHAP-weighted priority |
| 🔬 Model Insights | Global SHAP summary plots and feature importance |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/Balajis111/iot-ids-xgboost.git
cd iot-ids-xgboost
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Open in browser
```
http://localhost:8501
```

---

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| XGBoost | Binary classification |
| SHAP | Model explainability |
| Streamlit | Dashboard UI |
| Pandas / NumPy | Data processing |
| Scikit-learn | Preprocessing, metrics |
| Plotly | Interactive charts |
| Joblib | Model serialization |

---

## 🌿 Branches

| Branch | Description |
|--------|-------------|
| `main` | Original baseline model |
| `additional-features` | Real-time input + SHAP priority scoring + SOC dashboard |
| `paper-features` | Research paper implementation |

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `app.py` | Main dashboard — all 5 pages |
| `realtime_input.py` | Manual entry form + synthetic flow generator |
| `alert_priority.py` | SWAPS scoring engine + triage card renderer |
| `notebooks/01_eda.ipynb` | Full training pipeline |

---

## 👨‍💻 Author

**Balaji S**
B.E. Computer Science & Engineering (Cyber Security)
Sri Siddhartha Institute of Technology

---

## 📄 License

MIT License
