# Rule-Based Expert System for Network Troubleshooting (Hybrid ML)

A production-style Python expert system that combines:

- **Rule-based diagnosis** (knowledge base + forward chaining inference engine)
- **Machine learning classifier** (RandomForest)
- **Hybrid decision layer** (use strong rule conclusions first, then fallback to ML)

## Project Structure

```text
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│   ├── trained_model.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── rules.py
│   ├── inference_engine.py
│   ├── ml_model.py
│   ├── evaluator.py
│   ├── utils.py
│
├── app/
│   ├── main.py
│   ├── dashboard.py
│
├── notebooks/
│   ├── experimentation.ipynb
│
├── requirements.txt
└── README.md
```

## Supported Labels

- `normal`
- `slow_network`
- `packet_loss_issue`
- `congestion_issue`
- `security_attack`

## Dataset Format

Input CSV should contain the following columns:

- `latency`
- `packet_loss`
- `jitter`
- `bandwidth_usage`
- `congestion`
- `traffic_anomaly`
- `label`

An example file is provided at:

- `data/raw/network_metrics_sample.csv`

## How It Works

1. **Data Processing**
   - Load CSV
   - Impute missing values (median)
   - Normalize features (standard scaling)
   - Encode labels
   - Split train/test
2. **Rule-Based Core**
   - Rules defined as structured objects in `src/rules.py`
   - Forward chaining in `src/inference_engine.py`
   - Priority and confidence determine the best rule outcome
3. **ML Component**
   - Train RandomForest classifier
   - Save model to `models/trained_model.pkl`
4. **Hybrid Layer**
   - If rule confidence is above threshold (default `0.9`) use rules
   - Else fallback to ML prediction
5. **Evaluation**
   - Computes Accuracy, Precision, Recall, F1
   - Compares rule-based, ML, and hybrid

## Installation

### 1) Create virtual environment

```bash
python -m venv .venv
```

### 2) Activate it

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Install this project locally (recommended)

```bash
pip install -e .
```

> Do **not** run `pip install src`; that installs a different PyPI project.
>
> If `pip install -e .` says no `pyproject.toml` was found, verify you are in the repository root (same folder as `README.md`). A `setup.py` fallback is also included.

## Run CLI

```bash
python -m app.main --help
# or, after editable install:
network-expert --help
```

## Train + Evaluate

```bash
python -m app.main train --data data/raw/network_metrics_sample.csv
# or
network-expert train --data data/raw/network_metrics_sample.csv
```

## Run CLI Diagnosis

```bash
python -m app.main diagnose \
  --latency 110 \
  --packet_loss 1.2 \
  --jitter 20 \
  --bandwidth_usage 70 \
  --congestion 0.55 \
  --traffic_anomaly 0.3
```

## Open the UI (Streamlit)

```bash
streamlit run app/dashboard.py
```

Then browse to `http://localhost:8501`.

## Notes for Production Hardening

- Replace sample dataset with real telemetry pipelines
- Add model versioning and drift monitoring
- Add rule management UI and rule validation tests
- Add API service layer (FastAPI) and authentication
- Integrate with SIEM/NMS tools for automated remediation
