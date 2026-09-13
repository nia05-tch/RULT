import numpy as np
from scipy import stats
from typing import Dict

class CalibrationAnalyzer:
    """Check if predicted uncertainty matches empirical uncertainty."""
    
    @staticmethod
    def calibration_error(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        std_pred: np.ndarray,
        confidence: float = 0.95,
    ) -> Dict:
        """
        Measure Expected Calibration Error (ECE).
        
        Answers RQ2: Is uncertainty calibrated?
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        std_pred = np.asarray(std_pred).flatten()
        
        residuals = np.abs(y_true - y_pred)
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        observed_coverage = (residuals <= z_score * std_pred).mean()
        expected_coverage = confidence
        
        ece = abs(observed_coverage - expected_coverage)
        
        return {
            'observed_coverage': float(observed_coverage),
            'expected_coverage': float(expected_coverage),
            'calibration_error': float(ece),
            'is_calibrated': bool(ece < 0.1),
            'z_score_used': float(z_score),
            'confidence': confidence,
        }
