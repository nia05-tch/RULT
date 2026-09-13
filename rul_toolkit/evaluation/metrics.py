"""Standard RUL evaluation metrics."""

import numpy as np


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))) * 100


def phm_score(y_true, y_pred):
    """
    Asymmetric PHM scoring function used in many CMAPSS papers.
    Heavily penalizes late predictions.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    d = y_pred - y_true

    score = np.where(
        d < 0,
        np.exp(-d / 13) - 1,
        np.exp(d / 10) - 1,
    )

    return float(np.sum(score))


def scoring_summary(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "phm_score": phm_score(y_true, y_pred),
    }
