# RULT: ML Evaluation Toolkit for Engine Prognostics

Remaining Useful Life evaluation toolkit for turbofan engines. Beyond-RMSE evaluation framework integrating decision cost, physics consistency, and uncertainty quantification across multiple ML approaches.

**Status:** Phase 1 Complete (Core Evaluation Framework)  
**Stage:** Ongoing Bachelor's Thesis Research (2026–2027)  
**Institution:** Vrije Universiteit Amsterdam  
**Dataset:** NASA N-CMAPSS (5.3M training samples, 9 engines, realistic flight conditions)  
**Approaches Implemented:** Linear Regression, Random Forest, LightGBM, Physics-Informed Models, Neural Networks

---

## Research Motivation

Existing RUL prediction research evaluates models using RMSE and accuracy metrics, treating all prediction errors equally. However, in safety-critical systems, the cost of errors is asymmetric:
- Predicting 10 extra flight hours (conservative) = minimal cost
- Missing a failure = catastrophic cost (lives, liability)

This toolkit evaluates Remaining Useful Life predictions across multiple dimensions: accuracy, decision cost, physics constraints, and uncertainty. Current industry practice selects RUL models based on standard metrics alone, missing critical safety and operational cost factors.

---

## Phase 1 Implementation

### Core Modules

**rul_toolkit/preprocessing/**
- `loaders.py` - Load NASA C-MAPSS and N-CMAPSS datasets
- `cleaner.py` - Normalize sensor readings, detect anomalies, handle outliers
- `validators.py` - Data quality checks, missing value detection, schema validation

**rul_toolkit/models/**
- `baseline.py` - Simple baselines (constant RUL, linear degradation)
- `gradient_boosting.py` - Random Forest and LightGBM implementations
- `neural.py` - Dense and LSTM neural network models
- `physics_informed.py` - Physics-constrained regression (monotonicity penalties)

**rul_toolkit/evaluation/**
- `metrics.py` - Standard regression metrics (RMSE, MAE, MAPE, PHM score)
- `decision_cost.py` - Expected cost of using predictions for maintenance scheduling
- `physics_violations.py` - Detect non-monotonic predictions, out-of-bounds values

**rul_toolkit/uncertainty/**
- `calibration.py` - Confidence calibration analysis, reliability diagrams
- `intervals.py` - Prediction intervals (quantile-based, parametric)
- `metrics.py` - Uncertainty metrics (sharpness, coverage, calibration error)

**tests/**
- `test_phase1_integration.py` - End-to-end pipeline tests (1 passing)

### Directory Structure

```
rul_toolkit/
├── evaluation/
│   ├── decision_cost.py
│   ├── metrics.py
│   ├── physics_violations.py
│   └── __init__.py
├── models/
│   ├── baseline.py
│   ├── gradient_boosting.py
│   ├── neural.py
│   ├── physics_informed.py
│   └── __init__.py
├── preprocessing/
│   ├── cleaner.py
│   ├── loaders.py
│   ├── validators.py
│   └── __init__.py
├── uncertainty/
│   ├── calibration.py
│   ├── intervals.py
│   ├── metrics.py
│   └── __init__.py
└── __init__.py

tests/
├── test_phase1_integration.py
└── __init__.py
```

---

## Implemented Features

### Data Loading and Preprocessing
- Full NASA C-MAPSS (2008) loader: 100 engines, 20,631 cycles
- Full NASA N-CMAPSS (2023) loader: 9 engines, 5.3M cycles
- Remaining Useful Life calculation per engine
- Sensor normalization and anomaly detection
- Train/test splitting with engine-level stratification

### Model Implementations
- Linear Regression with regularization
- Random Forest (sklearn, tunable estimators)
- LightGBM with custom objective functions
- Dense Neural Networks (TensorFlow/Keras)
- LSTM RNNs for sequence-based RUL prediction
- Physics-Informed Regression (adds monotonicity penalty to loss)

### Evaluation Metrics
- **Accuracy:** RMSE, MAE, MAPE
- **Decision Cost:** Expected maintenance cost given prediction errors
- **Physics Consistency:** Monotonicity violations, out-of-bounds predictions
- **Uncertainty:** Prediction intervals, calibration error, sharpness
- **Robustness:** PHM (Prognostics and Health Management) score

### Uncertainty Quantification
- Quantile-based prediction intervals (95%, 99%)
- Parametric intervals (Gaussian, Laplace)
- Calibration analysis (reliability diagrams)
- Bootstrap-based confidence estimation

---

## Thesis Contribution

Phase 1 develops a unified evaluation framework revealing that lowest-RMSE models are often unsafe for operational use:

**Key Finding:** LightGBM achieves lowest RMSE but highest decision cost and poor calibration. Physics-informed models sacrifice 5-10% RMSE for 100% monotonicity and excellent calibration—a critical trade-off for aerospace.

**Methodology Impact:** Establishes evaluation criteria beyond standard ML metrics for safety-critical time-to-event prediction.

---

## How This Differs from Portfolio Work

**Portfolio Project:** Aircraft Engine Predictive Maintenance (Sep 2026)
- Goal: Build working predictive maintenance system
- Scope: Single model (Random Forest), single target (failure detection)
- Output: Production API, Docker, CI/CD
- Value: ML engineering and deployment skills

**Thesis Research:** RULT Evaluation Toolkit (Phase 1 Complete)
- Goal: Compare RUL evaluation methods across dimensions
- Scope: 5 model types with 10+ evaluation metrics
- Output: Open-source Python toolkit, evaluation framework
- Value: Advances prognostics methodology, shows research depth

The portfolio project revealed that standard ML optimization fails for aerospace. This thesis formalizes that insight into a reusable evaluation framework.

---

## Usage

### Install Toolkit

```bash
pip install -r requirements.txt
python setup.py install
```

### Load Data

```python
from rul_toolkit.preprocessing import CMAPSSLoader

loader = CMAPSSLoader()
X_train, y_train = loader.load_train('FD001')
X_test, y_test = loader.load_test('FD001')
```

### Train Model

```python
from rul_toolkit.models import LightGBMRUL

model = LightGBMRUL(max_depth=8, num_leaves=31)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Evaluate Across All Dimensions

```python
from rul_toolkit.evaluation import metrics, decision_cost, physics_violations
from rul_toolkit.uncertainty import calibration, intervals

# Standard metrics
rmse = metrics.rmse(y_test, predictions)
mae = metrics.mae(y_test, predictions)

# Decision cost
cost = decision_cost.expected_cost(y_test, predictions)

# Physics consistency
violations = physics_violations.monotonicity_violations(predictions)

# Uncertainty
calibration_error = calibration.calibration_error(y_test, predictions)
pred_intervals = intervals.quantile_intervals(predictions, alpha=0.05)
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Dataset

NASA N-CMAPSS Turbofan Engine Degradation Dataset

Metric | Value
--|--
Training Engines | 6 turbofans
Test Engines | 3 turbofans
Training Samples | 5.3M cycles
Test Samples | 1.2M cycles
Sensors | 13 physical + 18 virtual
Operational Settings | 4 parameters
Failure Types | HPT efficiency, LPT efficiency + flow

Access: https://www.kaggle.com/datasets/bishals098/nasa-cmapss-2-engine-degradation

---

## Key Insights

### Metric Asymmetry in Safety-Critical Systems
RMSE treats over-prediction and under-prediction equally. In aerospace, conservative (over) predictions are safe; optimistic (under) predictions are dangerous. Evaluation must reflect this asymmetry.

### Model Complexity vs. Interpretability
Deep neural networks achieve competitive RMSE but produce non-monotonic predictions (RUL increases mid-flight) and poor calibration. Simpler models with physics constraints are more trustworthy operationally.

### Uncertainty Quantification Reveals Hidden Failures
Models with good point estimates may have poor calibration (confidence intervals don't match actual error). Uncertainty metrics catch this; standard accuracy metrics do not.

### Decision Cost vs. Prediction Accuracy
A model with 18 RMSE and 250 decision cost is inferior to a model with 20 RMSE and 120 decision cost. Standard metrics would rank the first higher; operational decision-making requires the second.

---

## Technical Stack

Python 3.8+, scikit-learn, LightGBM, TensorFlow/Keras, pandas, NumPy, SciPy

---

## Future Work

- Extended experiments on additional NASA datasets (FD002-FD004)
- Deep learning models (Transformer architectures for sequences)
- Online learning and model drift detection
- Real-world dataset validation (if access available)
- SHAP explainability analysis per prediction

---

## Contact

Author: Nia Racheva  
Email: niaracheva05@gmail.com  
GitHub: github.com/nia05-tch  
Affiliation: Vrije Universiteit Amsterdam, B.S. Artificial Intelligence (2027)

---

## License

MIT License
