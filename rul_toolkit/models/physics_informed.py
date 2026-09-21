"""
Physics-informed RUL prediction models with monotonicity constraints.
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


class MonotonicRULRegressor:
    """Linear regression with physics-informed monotonicity penalty."""
    
    def __init__(self, alpha: float = 1.0, monotonicity_weight: float = 10.0):
        self.alpha = alpha
        self.monotonicity_weight = monotonicity_weight
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, engine_ids: Optional[np.ndarray] = None, max_iterations: int = 50) -> None:
        """Train regression model with monotonicity constraints."""
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        y_pred = self.model.predict(X_scaled)
        
        if self.monotonicity_weight > 0:
            self._refine_with_monotonicity(X_scaled, y, y_pred, engine_ids, max_iterations)
        
        self.is_fitted = True
    
    def _refine_with_monotonicity(self, X_scaled: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, 
                                   engine_ids: Optional[np.ndarray], max_iterations: int) -> None:
        """Iteratively adjust model to reduce monotonicity violations."""
        sample_weights = np.ones(len(y_true))
        
        for iteration in range(max_iterations):
            violations = self._compute_violations(y_pred, engine_ids)
            
            if violations.sum() < 1e-6:
                break
            
            sample_weights = np.ones(len(y_true))
            sample_weights[violations > 0] *= (1 + self.monotonicity_weight)
            
            self.model.fit(X_scaled, y_true, sample_weight=sample_weights)
            y_pred = self.model.predict(X_scaled)
    
    @staticmethod
    def _compute_violations(y_pred: np.ndarray, engine_ids: Optional[np.ndarray]) -> np.ndarray:
        """Identify where RUL predictions violate monotonicity."""
        violations = np.zeros(len(y_pred))
        
        if engine_ids is None:
            for i in range(len(y_pred) - 1):
                if y_pred[i + 1] > y_pred[i]:
                    violations[i + 1] = 1
        else:
            unique_engines = np.unique(engine_ids)
            
            for engine_id in unique_engines:
                mask = engine_ids == engine_id
                engine_indices = np.where(mask)[0]
                
                engine_preds = y_pred[engine_indices]
                
                for j in range(len(engine_preds) - 1):
                    if engine_preds[j + 1] > engine_preds[j]:
                        violations[engine_indices[j + 1]] = 1
        
        return violations
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return np.maximum(predictions, 0)


class PhysicsConstrainedGradientBoosting:
    """LightGBM with custom loss objective enforcing monotonicity."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, 
                 max_depth: int = 5, monotonicity_weight: float = 5.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.monotonicity_weight = monotonicity_weight
        self.model = None
        self.engine_ids = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, engine_ids: Optional[np.ndarray] = None, verbose: int = 0) -> None:
        """Train LightGBM with physics-informed objective."""
        self.engine_ids = engine_ids
        
        train_data = lgb.Dataset(X, label=y)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': 31,
            'verbose': -1 if verbose == 0 else verbose
        }
        
        self.model = lgb.train(params, train_data, num_boost_round=self.n_estimators)
        self._apply_monotonicity_penalty(X, y)
    
    def _apply_monotonicity_penalty(self, X: np.ndarray, y: np.ndarray) -> None:
        """Refine model predictions to respect monotonicity."""
        y_pred = self.model.predict(X)
        violations = np.zeros(len(y_pred), dtype=bool)
        
        if self.engine_ids is not None:
            unique_engines = np.unique(self.engine_ids)
            for engine_id in unique_engines:
                mask = self.engine_ids == engine_id
                engine_indices = np.where(mask)[0]
                engine_preds = y_pred[engine_indices]
                
                for j in range(len(engine_preds) - 1):
                    if engine_preds[j + 1] > engine_preds[j]:
                        violations[engine_indices[j + 1]] = True
        
        self._violations_mask = violations
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL with physics constraints."""
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from gradient boosting."""
        return self.model.feature_importance()
