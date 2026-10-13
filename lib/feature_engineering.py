import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Advanced feature engineering for financial time series data
    """
    
    def __init__(self):
        self.feature_names = []
        self.scaling_params = {}
        
    def create_lagged_features(self, data: pd.DataFrame, lags: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """Create lagged versions of all features"""
        lagged_data = data.copy()
        
        for col in data.columns:
            for lag in lags:
                lagged_col = f"{col}_lag_{lag}"
                lagged_data[lagged_col] = data[col].shift(lag)
                self.feature_names.append(lagged_col)
                
        return lagged_data
    
    def create_rate_of_change_features(self, data: pd.DataFrame, periods: List[int] = [1, 3, 6]) -> pd.DataFrame:
        """Create rate of change (momentum) features"""
        roc_data = data.copy()
        
        for col in data.columns:
            if not col.endswith(('_lag_3', '_lag_6', '_lag_12')):  # Don't create ROC for lag features
                for period in periods:
                    roc_col = f"{col}_roc_{period}"
                    roc_data[roc_col] = data[col].pct_change(periods=period)
                    self.feature_names.append(roc_col)
                    
        return roc_data
    
    def create_moving_average_features(self, data: pd.DataFrame, windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """Create moving average features and ratios"""
        ma_data = data.copy()
        
        for col in data.columns:
            if not any(suffix in col for suffix in ['_lag_', '_roc_']):  # Don't create MA for derived features
                for window in windows:
                    # Moving average
                    ma_col = f"{col}_ma_{window}"
                    ma_data[ma_col] = data[col].rolling(window=window).mean()
                    self.feature_names.append(ma_col)
                    
                    # Ratio to moving average (mean reversion indicator)
                    ratio_col = f"{col}_to_ma_{window}"
                    ma_data[ratio_col] = data[col] / ma_data[ma_col]
                    self.feature_names.append(ratio_col)
                    
        return ma_data
    
    def create_volatility_features(self, data: pd.DataFrame, windows: List[int] = [6, 12]) -> pd.DataFrame:
        """Create volatility (rolling standard deviation) features"""
        vol_data = data.copy()
        
        for col in data.columns:
            if not any(suffix in col for suffix in ['_lag_', '_roc_', '_ma_', '_to_ma_']):
                for window in windows:
                    vol_col = f"{col}_vol_{window}"
                    vol_data[vol_col] = data[col].rolling(window=window).std()
                    self.feature_names.append(vol_col)
                    
        return vol_data
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction terms between key economic indicators"""
        interaction_data = data.copy()
        
        # Define key interaction pairs
        interaction_pairs = [
            ('Fed_Rate', 'Gold_Price'),
            ('Fed_Rate', 'Dollar_Index'),
            ('Dollar_Index', 'Oil_Price'),
            ('Gold_Price', 'Silver_Price'),
            ('Oil_Price', 'Silver_Price')
        ]
        
        for var1, var2 in interaction_pairs:
            if var1 in data.columns and var2 in data.columns:
                interaction_col = f"{var1}_x_{var2}"
                interaction_data[interaction_col] = data[var1] * data[var2]
                self.feature_names.append(interaction_col)
                
        return interaction_data
    
    def create_ratio_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create economic ratio features"""
        ratio_data = data.copy()
        
        # Define meaningful economic ratios
        ratio_pairs = [
            ('Gold_Price', 'Silver_Price', 'Gold_Silver_Ratio'),
            ('Oil_Price', 'Gold_Price', 'Oil_Gold_Ratio'),
            ('Dollar_Index', 'Gold_Price', 'Dollar_Gold_Ratio')
        ]
        
        for numerator, denominator, ratio_name in ratio_pairs:
            if numerator in data.columns and denominator in data.columns:
                ratio_data[ratio_name] = data[numerator] / data[denominator]
                self.feature_names.append(ratio_name)
                
        return ratio_data
    
    def create_polynomial_features(self, data: pd.DataFrame, degree: int = 2, 
                                 base_features: Optional[List[str]] = None) -> pd.DataFrame:
        """Create polynomial features for base economic indicators"""
        poly_data = data.copy()
        
        if base_features is None:
            base_features = ['Fed_Rate', 'Dollar_Index', 'Gold_Price', 'Oil_Price', 'Silver_Price']
        
        # Only create polynomial features for base indicators to avoid explosion
        for col in base_features:
            if col in data.columns:
                for deg in range(2, degree + 1):
                    poly_col = f"{col}_poly_{deg}"
                    poly_data[poly_col] = data[col] ** deg
                    self.feature_names.append(poly_col)
                    
        return poly_data
    
    def create_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal/cyclical features"""
        seasonal_data = data.copy()
        
        # Month-based features
        seasonal_data['Month'] = data.index.month
        seasonal_data['Quarter'] = data.index.quarter
        seasonal_data['Month_Sin'] = np.sin(2 * np.pi * seasonal_data['Month'] / 12)
        seasonal_data['Month_Cos'] = np.cos(2 * np.pi * seasonal_data['Month'] / 12)
        
        self.feature_names.extend(['Month', 'Quarter', 'Month_Sin', 'Month_Cos'])
        
        return seasonal_data
    
    def engineer_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations"""
        print("\n=== Feature Engineering ===")
        print(f"Starting with {len(data.columns)} features")
        
        # Reset feature names for fresh run
        self.feature_names = []
        
        # Apply transformations in sequence
        engineered_data = data.copy()
        
        print("Creating lagged features...")
        engineered_data = self.create_lagged_features(engineered_data)
        
        print("Creating rate of change features...")
        engineered_data = self.create_rate_of_change_features(engineered_data)
        
        print("Creating moving average features...")
        engineered_data = self.create_moving_average_features(engineered_data)
        
        print("Creating volatility features...")
        engineered_data = self.create_volatility_features(engineered_data)
        
        print("Creating interaction features...")
        engineered_data = self.create_interaction_features(engineered_data)
        
        print("Creating ratio features...")
        engineered_data = self.create_ratio_features(engineered_data)
        
        print("Creating polynomial features...")
        engineered_data = self.create_polynomial_features(engineered_data)
        
        print("Creating seasonal features...")
        engineered_data = self.create_seasonal_features(engineered_data)
        
        print(f"Feature engineering complete: {len(engineered_data.columns)} total features")
        print(f"Added {len(engineered_data.columns) - len(data.columns)} new features")
        
        return engineered_data
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Group features by type for analysis"""
        groups = {
            'base': [],
            'lagged': [],
            'momentum': [],
            'moving_average': [],
            'volatility': [],
            'interaction': [],
            'ratio': [],
            'polynomial': [],
            'seasonal': []
        }
        
        for feature in self.feature_names:
            if '_lag_' in feature:
                groups['lagged'].append(feature)
            elif '_roc_' in feature:
                groups['momentum'].append(feature)
            elif '_ma_' in feature or '_to_ma_' in feature:
                groups['moving_average'].append(feature)
            elif '_vol_' in feature:
                groups['volatility'].append(feature)
            elif '_x_' in feature:
                groups['interaction'].append(feature)
            elif 'Ratio' in feature:
                groups['ratio'].append(feature)
            elif '_poly_' in feature:
                groups['polynomial'].append(feature)
            elif feature in ['Month', 'Quarter', 'Month_Sin', 'Month_Cos']:
                groups['seasonal'].append(feature)
            else:
                groups['base'].append(feature)
                
        return groups