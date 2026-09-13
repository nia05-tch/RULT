import numpy as np

from rul_toolkit.evaluation.metrics import (
    rmse,
    mae,
    mape,
    phm_score,
)
from rul_toolkit.evaluation.decision_cost import expected_decision_cost
from rul_toolkit.evaluation.physics_violations import monotonicity_violation_rate

from rul_toolkit.preprocessing.cleaner import DataCleaner

from rul_toolkit.models.baseline import SimpleRUL, ConstantRUL
from rul_toolkit.models.gradient_boosting import RandomForestRUL, LightGBMRUL

from rul_toolkit.uncertainty.intervals import quantile_prediction_intervals
from rul_toolkit.uncertainty.calibration import CalibrationAnalyzer


def test_phase1_end_to_end():
    """Basic end-to-end smoke test for the Phase 1 toolkit."""

    rng = np.random.default_rng(42)

    # Synthetic regression dataset
    X = rng.normal(size=(100, 5))
    y = 50 - 3 * X[:, 0] + 2 * X[:, 1] + rng.normal(0, 1, 100)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    predictions = y + rng.normal(0, 2, 100)

    assert np.isfinite(rmse(y, predictions))
    assert np.isfinite(mae(y, predictions))
    assert np.isfinite(mape(y, predictions))
    assert np.isfinite(phm_score(y, predictions))

    cost = expected_decision_cost(y, predictions)
    assert np.isfinite(cost.total_cost)

    # Chronological sequence: RUL should not increase
    true_sequence = np.array([50, 45, 40, 35, 30, 25, 20, 15, 10, 5])
    predicted_sequence = np.array([49, 44, 41, 34, 29, 26, 19, 16, 9, 4])

    violation_rate = monotonicity_violation_rate(predicted_sequence)
    assert np.isfinite(violation_rate["violation_rate"])

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------
    cleaner = DataCleaner()

    import pandas as pd

    X_df = pd.DataFrame(X, columns=[f"sensor_{i}" for i in range(X.shape[1])])

    X_clean, scaler, columns = cleaner.clean_pipeline(
        X_df,
        remove_anomalies=False,
    )

    assert X_clean.shape == X.shape
    assert np.all(np.isfinite(X_clean))
    assert len(columns) == X.shape[1]
    assert scaler is not None

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------
    models = [
        SimpleRUL(),
        ConstantRUL(),
        RandomForestRUL(
            n_estimators=20,
            random_state=42,
        ),
        LightGBMRUL(
            n_estimators=20,
            random_state=42,
        ),
    ]

    for model in models:
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert np.all(np.isfinite(predictions))

    # ---------------------------------------------------------
    # Prediction intervals
    # ---------------------------------------------------------
    residuals = y - predictions

    intervals = quantile_prediction_intervals(
        predictions,
        residuals,
        confidence=0.90,
    )

    lower = np.asarray(intervals["lower_bounds"])
    upper = np.asarray(intervals["upper_bounds"])

    assert lower.shape == predictions.shape
    assert upper.shape == predictions.shape
    assert np.all(lower <= upper)

    # ---------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------
    analyzer = CalibrationAnalyzer()

    std_pred = np.full_like(predictions, np.std(residuals))

    calibration = analyzer.calibration_error(
        y,
        predictions,
        std_pred,
        confidence=0.90,
    )

    assert np.isfinite(calibration["calibration_error"])
    assert 0.0 <= calibration["observed_coverage"] <= 1.0

    print("\nPhase 1 integration test passed.")


if __name__ == "__main__":
    test_phase1_end_to_end()
