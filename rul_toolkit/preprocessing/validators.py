import numpy as np
import pandas as pd

def check_data_quality(df: pd.DataFrame) -> dict:
    """Comprehensive data quality report."""
    report = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'n_missing': df.isnull().sum().sum(),
        'missing_pct': float(100 * df.isnull().sum().sum() / (len(df) * len(df.columns))),
    }
    
    missing_by_col = df.isnull().sum()
    if missing_by_col.sum() > 0:
        report['missing_by_column'] = missing_by_col.to_dict()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report['numeric_stats'] = {}
    
    for col in numeric_cols:
        report['numeric_stats'][col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
        }
    
    return report


def verify_rul_column(y: pd.Series) -> dict:
    """Verify RUL target is valid."""
    return {
        'n_samples': len(y),
        'min_rul': float(y.min()),
        'max_rul': float(y.max()),
        'mean_rul': float(y.mean()),
        'std_rul': float(y.std()),
        'has_negatives': bool((y < 0).any()),
        'has_zeros': bool((y == 0).any()),
    }


def train_test_split_unit_wise(
    X: pd.DataFrame,
    y: pd.Series,
    engine_ids: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split data such that same engine doesn't appear in train and test."""
    np.random.seed(random_state)
    
    unique_engines = engine_ids.unique()
    n_test_engines = max(1, int(len(unique_engines) * test_size))
    
    test_engines = np.random.choice(unique_engines, size=n_test_engines, replace=False)
    
    test_mask = engine_ids.isin(test_engines)
    train_mask = ~test_mask
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    return X_train, X_test, y_train, y_test
