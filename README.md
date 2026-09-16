# RULT: ML Evaluation Toolkit for Engine Prognostics

Python toolkit for evaluating Remaining Useful Life (RUL) models on turbofan engines.  
Goes beyond standard RMSE by also measuring decision cost, physical consistency, and uncertainty.

Status: Phase 1 complete — toolkit is implemented and tested. Large-scale experiments are next.  
Stage: Ongoing Bachelor’s thesis (2026–2027)  
Institution: Vrije Universiteit Amsterdam  
Dataset: NASA N-CMAPSS (5.3M training samples, 9 engines)

## Why this exists

Most RUL papers only report RMSE or accuracy. In real aerospace systems that is not enough:

- Predicting a bit too early (extra maintenance) is cheap
- Missing a failure is extremely expensive

This toolkit evaluates models on four axes at the same time: accuracy, decision cost, physics constraints, and uncertainty quantification.

## What is already implemented (Phase 1)

Preprocessing
- Loaders for both C-MAPSS and N-CMAPSS
- Sensor normalization, anomaly detection, quality checks
- Engine-level train/test split

Models
- Linear Regression
- Random Forest
- LightGBM
- Dense Neural Network + LSTM
- Physics-informed regression (monotonicity penalty)

Evaluation
- Standard metrics (RMSE, MAE, MAPE, PHM score)
- Decision-cost metric (cost of acting on the prediction)
- Physics violations (non-monotonic RUL, out-of-bounds values)
- Uncertainty tools (prediction intervals, calibration, coverage)

Tests
- Integration test covering the full pipeline (currently 1/1 passing)

## How to use

pip install -r requirements.txt
python setup.py install

from rul_toolkit.preprocessing import CMAPSSLoader
from rul_toolkit.models import LightGBMRUL
from rul_toolkit.evaluation import metrics, decision_cost

loader = CMAPSSLoader()
X_train, y_train = loader.load_train("FD001")
X_test, y_test = loader.load_test("FD001")

model = LightGBMRUL()
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(metrics.rmse(y_test, preds))
print(decision_cost.expected_cost(y_test, preds))

## Relation to my earlier project

The portfolio project aircraft-engine-predictive-maintenance showed that optimizing only for accuracy is not enough for safety-critical systems.  
RULT turns that observation into a reusable evaluation framework.

## Next steps (Phase 2)

- Run the full comparison of all models on N-CMAPSS
- Add more uncertainty methods
- Possible Transformer-based models later

## Author

Nia Racheva  
Email: niaracheva05@gmail.com  
GitHub: github.com/nia05-tch  
License: MIT
