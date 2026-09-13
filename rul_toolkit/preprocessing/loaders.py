import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

class CMAPSSLoader:
    """Load NASA N-CMAPSS turbofan degradation dataset."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    def load_dataset(self, subset: str = 'DS02', split: str = 'train') -> Tuple[pd.DataFrame, pd.Series]:
        """Load one dataset subset."""
        if split == 'train':
            prefix = f'train_{subset}'
        elif split == 'test':
            prefix = f'test_{subset}'
        else:
            raise ValueError("split must be 'train' or 'test'")
        
        files = list(self.data_dir.glob(f'*{prefix}*.csv'))
        
        if not files:
            raise FileNotFoundError(f"No CSV files found for {prefix}")
        
        data = pd.read_csv(files[0])
        
        if 'RUL' not in data.columns:
            data.rename(columns={data.columns[-1]: 'RUL'}, inplace=True)
        
        X = data.drop('RUL', axis=1)
        y = data['RUL']
        
        return X, y
    
    def load_all_subsets(self, split: str = 'train') -> Dict:
        """Load all 4 dataset subsets."""
        data = {}
        
        for subset in ['DS01', 'DS02', 'DS03', 'DS04']:
            try:
                X, y = self.load_dataset(subset=subset, split=split)
                data[subset] = {'X': X, 'y': y}
            except FileNotFoundError:
                print(f"Warning: {subset} not found")
        
        return data
    
    def get_sensor_columns(self, data: pd.DataFrame) -> list:
        """Identify sensor columns."""
        return [col for col in data.columns if 'W' in col or 'X' in col]


def load_cmapss(data_dir: str, subset: str = 'DS02', split: str = 'train'):
    """Simple interface to load N-CMAPSS data."""
    loader = CMAPSSLoader(data_dir)
    return loader.load_dataset(subset=subset, split=split)
