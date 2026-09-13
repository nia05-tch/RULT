import numpy as np
from scipy import stats
from typing import Dict

def quantile_prediction_intervals(
    y_pred: np.ndarray,
    y_true_residuals: np.ndarray,
    confidence: float = 0.95,
) -> Dict:
    """Create prediction intervals from empirical residuals."""
    y_pred = np.asarray(y_pred).flatten()
    y_true_residuals = np.asarray(y_true_residuals).flatten()
    
    alpha = (1 - confidence) / 2
    lower_quantile = np.quantile(y_true_residuals, alpha)
    upper_quantile = np.quantile(y_true_residuals, 1 - alpha)
    
    lower = y_pred + lower_quantile
    upper = y_pred + upper_quantile
    
    return {
        'predictions': y_pred.tolist(),
        'lower_bounds': lower.tolist(),
        'upper_bounds': upper.tolist(),
        'lower_quantile': float(lower_quantile),
        'upper_quantile': float(upper_quantile),
        'confidence': confidence,
        'width': float(upper_quantile - lower_quantile),
    }


def parametric_prediction_intervals(
    y_pred: np.ndarray,
    y_true_residuals: np.ndarray,
    confidence: float = 0.95,
) -> Dict:
    """Create prediction intervals assuming Gaussian residuals."""
    y_pred = np.asarray(y_pred).flatten()
    y_true_residuals = np.asarray(y_true_residuals).flatten()
    
    sigma = np.std(y_true_residuals)
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    margin = z_score * sigma
    lower = y_pred - margin
    upper = y_pred + margin
    
    return {
        'predictions': y_pred.tolist(),
        'lower_bounds': lower.tolist(),
        'upper_bounds': upper.tolist(),
        'std_residuals': float(sigma),
        'margin': float(margin),
        'confidence': confidence,
        'z_score': float(z_score),
    }
