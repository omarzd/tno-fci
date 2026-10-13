import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelSuite:
    """
    Suite of advanced machine learning models for financial conditions index prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.performance_metrics = {}
        self.feature_importance = {}
        
    def initialize_models(self):
        """Initialize all model types with optimized hyperparameters"""
        
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ),
            'svr_rbf': SVR(
                kernel='rbf',
                C=100,
                gamma='scale',
                epsilon=0.1
            ),
            'svr_linear': SVR(
                kernel='linear',
                C=1.0,
                epsilon=0.1
            ),
            'ridge': Ridge(
                alpha=1.0,
                random_state=42
            ),
            'lasso': Lasso(
                alpha=0.1,
                random_state=42,
                max_iter=2000
            ),
            'elastic_net': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42,
                max_iter=2000
            )
        }
        
        # Initialize scalers for each model (SVR requires scaling)
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
            
    def prepare_data_for_modeling(self, features: pd.DataFrame, target: pd.Series, 
                                test_size_months: int = 24) -> tuple:
        """Prepare data with proper train/test split for time series"""
        
        # Align features and target
        common_idx = features.dropna().index.intersection(target.dropna().index)
        features_clean = features.loc[common_idx]
        target_clean = target.loc[common_idx]
        
        # Time series split - use last test_size_months for testing
        split_idx = len(common_idx) - test_size_months
        train_idx = common_idx[:split_idx]
        test_idx = common_idx[split_idx:]
        
        X_train = features_clean.loc[train_idx]
        X_test = features_clean.loc[test_idx]
        y_train = target_clean.loc[train_idx]
        y_test = target_clean.loc[test_idx]
        
        print(f"\nData preparation:")
        print(f"Total samples: {len(common_idx)}")
        print(f"Training samples: {len(X_train)} ({train_idx[0]:%Y-%m} to {train_idx[-1]:%Y-%m})")
        print(f"Testing samples: {len(X_test)} ({test_idx[0]:%Y-%m} to {test_idx[-1]:%Y-%m})")
        print(f"Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def fit_all_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit all models on training data"""
        print("\n=== Training Models ===")
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Scale features for models that need it
                if model_name.startswith('svr') or model_name in ['ridge', 'lasso', 'elastic_net']:
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
                
                print(f"✓ {model_name} trained successfully")
                
                # Store feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = pd.Series(
                        model.feature_importances_, 
                        index=X_train.columns
                    ).sort_values(ascending=False)
                    
            except Exception as e:
                print(f"✗ {model_name} failed: {e}")
                
    def predict_all_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series):
        """Generate predictions for all models"""
        print("\n=== Generating Predictions ===")
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict'):  # Check if model was fitted successfully
                try:
                    # Scale features if needed
                    if model_name.startswith('svr') or model_name in ['ridge', 'lasso', 'elastic_net']:
                        X_train_scaled = self.scalers[model_name].transform(X_train)
                        X_test_scaled = self.scalers[model_name].transform(X_test)
                        
                        train_pred = model.predict(X_train_scaled)
                        test_pred = model.predict(X_test_scaled)
                    else:
                        train_pred = model.predict(X_train)
                        test_pred = model.predict(X_test)
                    
                    # Store predictions
                    self.predictions[model_name] = {
                        'train': pd.Series(train_pred, index=X_train.index),
                        'test': pd.Series(test_pred, index=X_test.index)
                    }
                    
                    # Calculate performance metrics
                    train_r2 = r2_score(y_train, train_pred)
                    test_r2 = r2_score(y_test, test_pred)
                    train_corr, _ = pearsonr(y_train, train_pred)
                    test_corr, _ = pearsonr(y_test, test_pred)
                    
                    self.performance_metrics[model_name] = {
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_correlation': train_corr,
                        'test_correlation': test_corr,
                        'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                        'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
                    }
                    
                    print(f"✓ {model_name:15} - Test R²: {test_r2:.4f}, Test Corr: {test_corr:.4f}")
                    
                except Exception as e:
                    print(f"✗ {model_name} prediction failed: {e}")
    
    def create_ensemble_predictions(self, method: str = 'average'):
        """Create ensemble predictions from individual models"""
        print(f"\n=== Creating {method.title()} Ensemble ===")
        
        if not self.predictions:
            print("No predictions available for ensemble")
            return
        
        # Get all model predictions
        train_preds = []
        test_preds = []
        model_names = []
        
        for model_name, preds in self.predictions.items():
            if 'train' in preds and 'test' in preds:
                train_preds.append(preds['train'])
                test_preds.append(preds['test'])
                model_names.append(model_name)
        
        if len(train_preds) == 0:
            print("No valid predictions for ensemble")
            return
        
        # Convert to DataFrame for easier manipulation
        train_df = pd.DataFrame(train_preds).T
        test_df = pd.DataFrame(test_preds).T
        train_df.columns = model_names
        test_df.columns = model_names
        
        if method == 'average':
            # Simple average ensemble
            ensemble_train = train_df.mean(axis=1)
            ensemble_test = test_df.mean(axis=1)
            
        elif method == 'weighted':
            # Weight by test R² performance
            weights = []
            for model_name in model_names:
                if model_name in self.performance_metrics:
                    # Use test R² as weight, but ensure non-negative
                    weight = max(0, self.performance_metrics[model_name]['test_r2'])
                    weights.append(weight)
                else:
                    weights.append(0)
            
            # Normalize weights
            if sum(weights) > 0:
                weights = np.array(weights) / sum(weights)
            else:
                weights = np.ones(len(weights)) / len(weights)
            
            ensemble_train = (train_df * weights).sum(axis=1)
            ensemble_test = (test_df * weights).sum(axis=1)
            
            print(f"Ensemble weights: {dict(zip(model_names, weights))}")
        
        # Store ensemble predictions
        self.predictions[f'ensemble_{method}'] = {
            'train': ensemble_train,
            'test': ensemble_test
        }
        
        return ensemble_train, ensemble_test
    
    def evaluate_ensemble(self, y_train: pd.Series, y_test: pd.Series, method: str = 'average'):
        """Evaluate ensemble performance"""
        ensemble_name = f'ensemble_{method}'
        
        if ensemble_name not in self.predictions:
            print(f"No {method} ensemble found")
            return
        
        train_pred = self.predictions[ensemble_name]['train']
        test_pred = self.predictions[ensemble_name]['test']
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_corr, _ = pearsonr(y_train, train_pred)
        test_corr, _ = pearsonr(y_test, test_pred)
        
        self.performance_metrics[ensemble_name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_correlation': train_corr,
            'test_correlation': test_corr,
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred))
        }
        
        print(f"✓ {ensemble_name:15} - Test R²: {test_r2:.4f}, Test Corr: {test_corr:.4f}")
        
        return test_r2, test_corr
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get comprehensive performance summary"""
        if not self.performance_metrics:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, metrics in self.performance_metrics.items():
            summary_data.append({
                'Model': model_name,
                'Train_R2': metrics['train_r2'],
                'Test_R2': metrics['test_r2'],
                'Train_Correlation': metrics['train_correlation'],
                'Test_Correlation': metrics['test_correlation'],
                'Train_RMSE': metrics['train_rmse'],
                'Test_RMSE': metrics['test_rmse']
            })
        
        df = pd.DataFrame(summary_data)
        return df.sort_values('Test_R2', ascending=False)
    
    def get_top_features(self, model_name: str, top_n: int = 20) -> pd.Series:
        """Get top features for a specific model"""
        if model_name in self.feature_importance:
            return self.feature_importance[model_name].head(top_n)
        else:
            return pd.Series()
    
    def cross_validate_models(self, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5):
        """Perform time series cross validation"""
        print(f"\n=== Cross Validation ({cv_splits} splits) ===")
        
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        cv_results = {}
        
        for model_name, model in self.models.items():
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_cv = X.iloc[train_idx]
                X_val_cv = X.iloc[val_idx]
                y_train_cv = y.iloc[train_idx]
                y_val_cv = y.iloc[val_idx]
                
                try:
                    # Scale if needed
                    if model_name.startswith('svr') or model_name in ['ridge', 'lasso', 'elastic_net']:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_cv)
                        X_val_scaled = scaler.transform(X_val_cv)
                        
                        model.fit(X_train_scaled, y_train_cv)
                        val_pred = model.predict(X_val_scaled)
                    else:
                        model.fit(X_train_cv, y_train_cv)
                        val_pred = model.predict(X_val_cv)
                    
                    score = r2_score(y_val_cv, val_pred)
                    cv_scores.append(score)
                    
                except Exception as e:
                    cv_scores.append(np.nan)
            
            cv_results[model_name] = {
                'mean_cv_score': np.nanmean(cv_scores),
                'std_cv_score': np.nanstd(cv_scores),
                'cv_scores': cv_scores
            }
            
            print(f"{model_name:15} - CV R²: {np.nanmean(cv_scores):.4f} ± {np.nanstd(cv_scores):.4f}")
        
        return cv_results