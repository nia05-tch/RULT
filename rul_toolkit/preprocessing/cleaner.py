import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class DataCleaner:
    """Clean and normalize sensor data."""
    
    def __init__(self):
        self.scaler = None
    
    def normalize_features(
        self,
        X: np.ndarray,
        fit_scaler: StandardScaler = None,
    ) -> Tuple[np.ndarray, StandardScaler]:
        """Standardize sensor readings to mean=0, std=1."""
        if fit_scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            self.scaler = fit_scaler
            X_scaled = fit_scaler.transform(X)
        
        return X_scaled, self.scaler
    
    def detect_anomalies(
        self,
        X: np.ndarray,
        threshold: float = 3.0,
    ) -> dict:
        """Detect outliers using z-score."""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        z_scores = np.abs((X - mean) / (std + 1e-8))
        anomaly_mask = (z_scores > threshold).any(axis=1)
        
        anomaly_indices = np.where(anomaly_mask)[0]
        
        return {
            'n_anomalies': int(anomaly_mask.sum()),
            'pct': float(100 * anomaly_mask.sum() / len(X)),
            'indices': anomaly_indices.tolist(),
        }
    
    def detect_stuck_sensors(self, X: pd.DataFrame, threshold: float = 0.01):
        """Find sensors with near-zero variance."""
        stuck = []
        
        for col in X.columns:
            if X[col].std() < threshold:
                stuck.append(col)
        
        return stuck
    
    def clean_pipeline(self, X, remove_anomalies=False, anomaly_threshold=3.0):
        """Full cleaning pipeline."""
        X_clean = X.copy()
        
        stuck = self.detect_stuck_sensors(X_clean)
        if stuck:
            X_clean = X_clean.drop(columns=stuck)
            print(f"Removed stuck sensors: {stuck}")
        
        if remove_anomalies:
            anomalies = self.detect_anomalies(X_clean.values, threshold=anomaly_threshold)
            clean_indices = np.setdiff1d(np.arange(len(X_clean)), anomalies['indices'])
            X_clean = X_clean.iloc[clean_indices]
            print(f"Removed {len(anomalies['indices'])} anomalous rows")
        
        X_normalized, scaler = self.normalize_features(X_clean.values)
        
        return X_normalized, scaler, X_clean.columns.tolist()
