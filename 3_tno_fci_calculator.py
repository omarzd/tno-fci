import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr
from scipy import stats
import argparse
import os
from datetime import datetime
import warnings
from util.archiver import create_archive_directory, archive_file, save_timestamp
# Try to import features - gracefully degrade if not available
try:
    from lib.feature_engineering import FeatureEngineer
    HAS_FEATURE_ENGINEERING = True
except ImportError:
    HAS_FEATURE_ENGINEERING = False
    print("⚠ Feature engineering module not available - using basic features only")

try:
    from lib.advanced_models import AdvancedModelSuite
    HAS_ADVANCED_MODELS = True
except ImportError:
    HAS_ADVANCED_MODELS = False
    print("⚠ Advanced models module not available - using linear model only")

warnings.filterwarnings('ignore')

class TNOFCICalculator:
    def __init__(self, data_dir='data', use_advanced_features=True, use_ml_models=True, output_dir='output'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.data = {}
        self.model_start_date = None
        self.correlation_start_date = None
        self.use_advanced_features = use_advanced_features and HAS_FEATURE_ENGINEERING
        self.use_ml_models = use_ml_models and HAS_ADVANCED_MODELS
        
        # Initialize feature engineering and ML components if available
        if self.use_advanced_features:
            self.feature_engineer = FeatureEngineer()
        if self.use_ml_models:
            self.ml_suite = AdvancedModelSuite()
            
        # Store results for comparison
        self.results_comparison = {}
        
    def load_data(self):
        """Load all data from local CSV files"""
        print("\n=== Loading Data ===\n")
        
        # Define data files and their required columns
        data_files = {
            'ism': ('clean_all/ismpmi_clean.csv', 'ISM_PMI'),
            'fed_rates': ('clean_all/fedrates_clean.csv', 'Fed_Rate'),
            'dxy': ('clean_all/dxy_clean.csv', 'Dollar_Index'),
            'gold': ('clean_all/gold_clean.csv', 'Gold_Price'),
            'oil': ('clean_all/oil_clean.csv', 'Oil_Price'),
            'silver': ('clean_all/silver_clean.csv', 'Silver_Price'),
        }
        
        # Load each data file
        for key, (filepath, column) in data_files.items():
            full_path = os.path.join(self.data_dir, filepath)
            if os.path.exists(full_path):
                try:
                    df = pd.read_csv(full_path, index_col=0, parse_dates=True)
                    if column in df.columns:
                        self.data[key] = df[[column]]
                    else:
                        # If specific column not found, use first column
                        self.data[key] = df[[df.columns[0]]]
                        self.data[key].columns = [column]
                    
                    print(f"✓ {key:12} loaded: {len(df)} records from {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")
                except Exception as e:
                    print(f"✗ {key:12} error: {e}")
            else:
                print(f"✗ {key:12} file not found: {full_path}")
        
            
    def determine_date_ranges(self):
        """Determine model building and correlation date ranges"""
        print("\n=== Determining Date Ranges ===\n")
        
        # Find earliest start date for each component needed for model building
        model_components = ['fed_rates', 'dxy', 'gold']  # Core components for model
        
        start_dates = {}
        for component in model_components:
            if component in self.data:
                start_dates[component] = self.data[component].index.min()
                
        if start_dates:
            # Model starts from the latest of the earliest dates (common start)
            self.model_start_date = max(start_dates.values())
            print(f"Model building will start from: {self.model_start_date:%Y-%m-%d}")
            print(f"  (Limited by: {[k for k,v in start_dates.items() if v == self.model_start_date]})")
        else:
            raise ValueError("No data available for model building")
            
        # Correlation calculation starts from when ISM data is available
        if 'ism' in self.data:
            ism_start = self.data['ism'].index.min()
            self.correlation_start_date = max(self.model_start_date, ism_start)
            print(f"\nCorrelation calculation will start from: {self.correlation_start_date:%Y-%m-%d}")
            if self.correlation_start_date > self.model_start_date:
                print(f"  (Limited by ISM data starting from {ism_start:%Y-%m-%d})")
        else:
            print("\n⚠ No ISM data available for correlation calculation")
            
    def prepare_monthly_data(self):
        """Resample all data to monthly frequency"""
        print("\n=== Preparing Monthly Data ===\n")
        
        monthly_data = {}
        
        for key, df in self.data.items():
            # Resample to monthly, using mean for the month
            monthly = df.resample('M').mean()
            
            # Forward fill any missing values within reasonable limits
            monthly = monthly.fillna(method='ffill', limit=3)
            
            monthly_data[key] = monthly
            print(f"✓ {key:12} resampled to monthly: {len(monthly)} records")
            
        self.monthly_data = monthly_data
        
    def optimize_lag_periods(self):
        """Optimize the lag period for best correlation"""
        print("\n=== Optimizing Lag Periods ===\n")
        
        if 'ism' not in self.monthly_data:
            print("No ISM data available for lag optimization")
            self.optimal_lag = 9  # Default
            return
        
        # Test different lag periods
        lag_periods = range(3, 19)  # Test 3 to 18 months
        correlations = {}
        
        print(f"Testing {len(lag_periods)} different lag periods...")
        
        # Combine model features
        model_features = pd.DataFrame()
        for key, df in self.monthly_data.items():
            if key != 'ism':
                model_features = model_features.join(df, how='outer')
        
        print(f"Normalizing {len(model_features.columns)} features...")
        
        # Normalize features
        normalized_features = pd.DataFrame()
        for col in model_features.columns:
            mean = model_features[col].mean()
            std = model_features[col].std()
            normalized_features[col] = (model_features[col] - mean) / std
        
        # Test each lag period
        ism_data = self.monthly_data['ism']['ISM_PMI']
        
        print("Optimizing for each lag period (this may take a few minutes)...")
        for i, lag in enumerate(lag_periods, 1):
            try:
                # Create forward ISM target
                ism_forward = ism_data.shift(-lag)
                
                # Align data
                start_date = self.correlation_start_date
                end_date = min(normalized_features.index.max(), ism_forward.dropna().index.max())
                
                features = normalized_features.loc[start_date:end_date]
                target = ism_forward.loc[start_date:end_date]
                
                # Get valid indices
                valid_idx = features.dropna().index.intersection(target.dropna().index)
                
                if len(valid_idx) > 50:  # Need sufficient data
                    features_valid = features.loc[valid_idx]
                    target_valid = target.loc[valid_idx]
                    
                    # Simple optimization for this lag
                    def objective(weights):
                        fci = np.dot(features_valid.values, weights)
                        correlation, _ = pearsonr(fci, target_valid.values)
                        return -correlation
                    
                    bounds = [(-1, 1) for _ in range(features_valid.shape[1])]
                    print(f"  [{i:2d}/{len(lag_periods)}] Optimizing lag {lag:2d} months...", end=" ", flush=True)
                    result = differential_evolution(objective, bounds, maxiter=200, seed=42)
                    
                    correlations[lag] = -result.fun
                    print(f"correlation = {-result.fun:.4f} (n={len(valid_idx)})")
                    
            except Exception as e:
                print(f"Lag {lag:2d} months: failed - {e}")
                continue
        
        if correlations:
            self.optimal_lag = max(correlations, key=correlations.get)
            print(f"\n✓ Optimal lag period: {self.optimal_lag} months (correlation: {correlations[self.optimal_lag]:.4f})")
        else:
            self.optimal_lag = 9  # Default fallback
            print("\n⚠ Using default lag of 9 months")
            
    def build_baseline_model(self):
        """Build baseline model using original methodology"""
        print("\n=== Building Baseline Model ===\n")
        
        # Combine all data from model start date
        combined_data = pd.DataFrame()
        
        for key, df in self.monthly_data.items():
            if key != 'ism':  # ISM is the target, not a component
                combined_data = combined_data.join(df, how='outer')
                
        # Filter to model start date
        combined_data = combined_data[combined_data.index >= self.model_start_date]
        
        # Drop any rows with missing values
        combined_data = combined_data.dropna()
        
        print(f"Baseline model data shape: {combined_data.shape}")
        print(f"Date range: {combined_data.index.min():%Y-%m-%d} to {combined_data.index.max():%Y-%m-%d}")
        
        # Normalize features (z-score)
        normalized_data = pd.DataFrame()
        for col in combined_data.columns:
            mean = combined_data[col].mean()
            std = combined_data[col].std()
            normalized_data[col] = (combined_data[col] - mean) / std
            
        self.baseline_model_data = combined_data
        self.baseline_normalized_data = normalized_data
        
        # Optimize weights for baseline
        if 'ism' in self.monthly_data and self.correlation_start_date:
            self.optimize_baseline_weights()
        else:
            # Use equal weights if no ISM data
            n_components = len(normalized_data.columns)
            self.baseline_weights = np.ones(n_components) / n_components
            print("⚠ No ISM data available - using equal weights")
            
    def optimize_baseline_weights(self):
        """Optimize weights for baseline model"""
        print("\n=== Optimizing Baseline Weights ===\n")
        
        # Get ISM data
        ism_data = self.monthly_data['ism']['ISM_PMI']
        
        # Create forward ISM using optimal lag
        ism_forward = ism_data.shift(-self.optimal_lag)
        
        # Align all data from correlation start date
        start_date = self.correlation_start_date
        end_date = min(self.baseline_normalized_data.index.max(), ism_forward.dropna().index.max())
        
        # Filter data to common period
        model_features = self.baseline_normalized_data.loc[start_date:end_date]
        ism_target = ism_forward.loc[start_date:end_date]
        
        # Drop any remaining NaN values
        valid_idx = model_features.dropna().index.intersection(ism_target.dropna().index)
        model_features = model_features.loc[valid_idx]
        ism_target = ism_target.loc[valid_idx]
        
        print(f"Baseline optimization using {len(valid_idx)} months of data")
        print(f"Date range: {valid_idx.min():%Y-%m-%d} to {valid_idx.max():%Y-%m-%d}")
        
        # Define objective function (minimize negative correlation)
        def objective(weights):
            fci = np.dot(model_features.values, weights)
            correlation, _ = pearsonr(fci, ism_target.values)
            return -correlation  # Negative because we minimize
            
        # Optimize
        n_vars = model_features.shape[1]
        bounds = [(-1, 1) for _ in range(n_vars)]
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=1000,
            popsize=50,
            seed=42
        )
        
        self.baseline_weights = result.x
        self.baseline_correlation = -result.fun
        
        print(f"\n✓ Baseline optimization complete!")
        print(f"  Best correlation with ISM {self.optimal_lag}m forward: {self.baseline_correlation:.4f}")
        print(f"\n  Baseline weights:")
        for i, col in enumerate(model_features.columns):
            print(f"    {col:15} {self.baseline_weights[i]:7.4f}")
            
        # Store baseline results
        self.results_comparison['baseline'] = {
            'correlation': self.baseline_correlation,
            'weights': dict(zip(model_features.columns, self.baseline_weights)),
            'n_features': len(model_features.columns)
        }
            
    def build_advanced_model(self):
        """Build model with feature engineering"""
        if not self.use_advanced_features:
            print("Advanced features disabled")
            return
            
        print("\n=== Building Enhanced Model ===\n")
        
        # Start with baseline normalized data
        base_features = self.baseline_normalized_data.copy()
        
        # Apply feature engineering
        if hasattr(self, 'feature_engineer'):
            features = self.feature_engineer.engineer_all_features(base_features)
            
            # Remove features with too many NaN values (>50% missing)
            missing_pct = features.isnull().sum() / len(features)
            valid_features = missing_pct[missing_pct < 0.5].index
            features = features[valid_features]
            
            # Fill remaining NaN values
            features = features.fillna(method='ffill').fillna(method='bfill')
            
            # Remove any remaining columns with NaN
            features = features.dropna(axis=1)
            
            print(f"Features shape: {features.shape}")
            print(f"Valid features after cleanup: {len(features.columns)}")
            
            self.features = features
            
            # Optimize model if we have ISM data
            if 'ism' in self.monthly_data:
                self.optimize_weights()
                
    def optimize_weights(self):
        """Optimize weights for model"""
        print("\n=== Optimizing Weights ===\n")
        
        # Get ISM data
        ism_data = self.monthly_data['ism']['ISM_PMI']
        
        # Create forward ISM using optimal lag
        ism_forward = ism_data.shift(-self.optimal_lag)
        
        # Align data
        start_date = self.correlation_start_date
        end_date = min(self.features.index.max(), ism_forward.dropna().index.max())
        
        # Filter data to common period
        model_features = self.features.loc[start_date:end_date]
        ism_target = ism_forward.loc[start_date:end_date]
        
        # Drop any remaining NaN values
        valid_idx = model_features.dropna().index.intersection(ism_target.dropna().index)
        model_features = model_features.loc[valid_idx]
        ism_target = ism_target.loc[valid_idx]
        
        print(f"Optimization using {len(valid_idx)} months of data")
        print(f"Features: {model_features.shape[1]}")
        
        # Use regularized optimization due to many features
        def objective(weights):
            fci = np.dot(model_features.values, weights)
            correlation, _ = pearsonr(fci, ism_target.values)
            # Add L2 regularization to prevent overfitting
            l2_penalty = 0.01 * np.sum(weights**2)
            return -correlation + l2_penalty
            
        # Optimize
        n_vars = model_features.shape[1]
        bounds = [(-1, 1) for _ in range(n_vars)]
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=1000,
            popsize=50,
            seed=42
        )
        
        self.weights = result.x
        
        # Calculate actual correlation (without regularization)
        fci = np.dot(model_features.values, self.weights)
        self.correlation, _ = pearsonr(fci, ism_target.values)
        
        print(f"\n✓ Optimization complete!")
        print(f"  Best correlation with ISM {self.optimal_lag}m forward: {self.correlation:.4f}")
        print(f"  Improvement over baseline: {self.correlation - self.baseline_correlation:.4f}")
        
        # Store linear results
        self.results_comparison['linear'] = {
            'correlation': self.correlation,
            'n_features': len(model_features.columns),
            'improvement': self.correlation - self.baseline_correlation
        }
        
        # Show top feature weights
        feature_weights = pd.Series(self.weights, index=model_features.columns)
        top_features = feature_weights.abs().sort_values(ascending=False).head(20)
        
        print(f"\n  Top 20 features by absolute weight:")
        for feature, weight in top_features.items():
            actual_weight = feature_weights[feature]
            print(f"    {feature[:30]:30} {actual_weight:7.4f}")
            
    def build_ml_models(self):
        """Build ML models"""
        if not self.use_ml_models or not hasattr(self, 'features'):
            print("ML models disabled or no features available")
            return
            
        print("\n=== Building ML Models ===\n")
        
        # Prepare target variable
        ism_data = self.monthly_data['ism']['ISM_PMI']
        ism_forward = ism_data.shift(-self.optimal_lag)
        
        # Align data
        start_date = self.correlation_start_date
        end_date = min(self.features.index.max(), ism_forward.dropna().index.max())
        
        X = self.features.loc[start_date:end_date]
        y = ism_forward.loc[start_date:end_date]
        
        # Drop any remaining NaN values
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        print(f"ML training data: {X.shape}")
        
        # Initialize and train models
        self.ml_suite.initialize_models()
        
        # Prepare train/test split
        X_train, X_test, y_train, y_test = self.ml_suite.prepare_data_for_modeling(X, y)
        
        # Fit all models
        self.ml_suite.fit_all_models(X_train, y_train)
        
        # Generate predictions
        self.ml_suite.predict_all_models(X_train, X_test, y_train, y_test)
        
        # Create ensemble predictions
        self.ml_suite.create_ensemble_predictions('average')
        self.ml_suite.create_ensemble_predictions('weighted')
        
        # Evaluate ensembles
        self.ml_suite.evaluate_ensemble(y_train, y_test, 'average')
        self.ml_suite.evaluate_ensemble(y_train, y_test, 'weighted')
        
        # Store ML results
        performance_summary = self.ml_suite.get_performance_summary()
        if not performance_summary.empty:
            best_model = performance_summary.iloc[0]
            self.results_comparison['best_ml'] = {
                'model_name': best_model['Model'],
                'test_r2': best_model['Test_R2'],
                'test_correlation': best_model['Test_Correlation'],
                'train_r2': best_model['Train_R2']
            }
            
            print(f"\n=== ML Model Performance Summary ===")
            print(performance_summary.head(10).to_string(index=False, float_format='%.4f'))
            
    def calculate_final_fci(self):
        """Calculate final FCI using best performing method"""
        print("\n=== Calculating Final FCI ===\n")
        
        # Choose best method based on results
        if hasattr(self, 'results_comparison'):
            best_method = 'baseline'
            best_correlation = self.baseline_correlation
            
            if 'linear' in self.results_comparison:
                if self.results_comparison['linear']['correlation'] > best_correlation:
                    best_method = 'linear'
                    best_correlation = self.results_comparison['linear']['correlation']
                    
            if 'best_ml' in self.results_comparison:
                if self.results_comparison['best_ml']['test_correlation'] > best_correlation:
                    best_method = 'ml'
                    best_correlation = self.results_comparison['best_ml']['test_correlation']
        
        print(f"Selected method: {best_method} (correlation: {best_correlation:.4f})")
        
        # Calculate FCI using selected method
        if best_method == 'baseline':
            fci_values = np.dot(self.baseline_normalized_data.values, self.baseline_weights)
            self.final_fci = pd.Series(fci_values, index=self.baseline_normalized_data.index, name='TNO_FCI')
            self.final_method = 'baseline'
            self.final_correlation = self.baseline_correlation
            
        elif best_method == 'linear':
            fci_values = np.dot(self.features.values, self.weights)
            self.final_fci = pd.Series(fci_values, index=self.features.index, name='TNO_FCI')
            self.final_method = 'linear'
            self.final_correlation = self.correlation
            
        elif best_method == 'ml':
            # Use best ML model predictions
            best_ml_name = self.results_comparison['best_ml']['model_name']
            if best_ml_name in self.ml_suite.predictions:
                train_pred = self.ml_suite.predictions[best_ml_name]['train']
                test_pred = self.ml_suite.predictions[best_ml_name]['test']
                
                # Combine train and test predictions
                all_pred = pd.concat([train_pred, test_pred]).sort_index()
                self.final_fci = pd.Series(all_pred.values, index=all_pred.index, name='TNO_FCI')
                self.final_method = f'ml_{best_ml_name}'
                self.final_correlation = self.results_comparison['best_ml']['test_correlation']
            
        print(f"✓ Final FCI calculated using {self.final_method}")
        print(f"  FCI range: {self.final_fci.min():.4f} to {self.final_fci.max():.4f}")
        print(f"  Records: {len(self.final_fci)}")
        
    def calculate_fci(self):
        """Calculate TNO FCI using optimized weights - for backward compatibility"""
        if hasattr(self, 'final_fci'):
            self.fci = self.final_fci
        else:
            # Fallback to baseline if final not calculated
            fci_values = np.dot(self.baseline_normalized_data.values, self.baseline_weights)
            self.fci = pd.Series(fci_values, index=self.baseline_normalized_data.index, name='TNO_FCI')
        
    def save_results(self):
        """Save results to CSV"""
        print("\n=== Saving Results ===\n")
        
        # Combine all data for output
        results = pd.DataFrame(self.final_fci)
        
        # Add original data
        if hasattr(self, 'baseline_model_data'):
            results = results.join(self.baseline_model_data, how='left')
        
        # Add ISM data if available
        if 'ism' in self.monthly_data:
            results = results.join(self.monthly_data['ism'])
            # Add forward ISM
            results[f'ISM_{self.optimal_lag}m_Forward'] = results['ISM_PMI'].shift(-self.optimal_lag)
            
        # Add baseline FCI for comparison if different from final
        if hasattr(self, 'baseline_weights') and self.final_method != 'baseline':
            baseline_fci = np.dot(self.baseline_normalized_data.values, self.baseline_weights)
            baseline_series = pd.Series(baseline_fci, index=self.baseline_normalized_data.index, name='Baseline_FCI')
            results = results.join(baseline_series, how='left')
        
        # Save main results file
        output_path = os.path.join(self.output_dir, 'fci_results.csv')
        os.makedirs(self.output_dir, exist_ok=True)
        results.to_csv(output_path)
        
        print(f"✓ Results saved to {output_path}")
        print(f"  Total records: {len(results)}")
        
        # Print recent values
        print(f"\n  Recent TNO FCI values:")
        display_cols = ['TNO_FCI', 'ISM_PMI']
        if f'ISM_{self.optimal_lag}m_Forward' in results.columns:
            display_cols.append(f'ISM_{self.optimal_lag}m_Forward')
        if 'Baseline_FCI' in results.columns:
            display_cols.append('Baseline_FCI')
            
        print(results[display_cols].tail(10))
        
        # Save model information
        self.save_model_info()
        
        # Save summary report
        self.save_summary_report()
        
        # Archive results
        self.archive_results()
        
    def save_model_info(self):
        """Save model metadata and comparison"""
        print("\n=== Saving Model Information ===")
        
        model_info = []
        model_info.append("TNO Financial Conditions Index Model")
        model_info.append("=" * 60)
        model_info.append("")
        
        # Date ranges
        model_info.append("Date Ranges:")
        model_info.append(f"- Model building start: {self.model_start_date:%Y-%m-%d}")
        if hasattr(self, 'correlation_start_date') and self.correlation_start_date:
            model_info.append(f"- Correlation calculation start: {self.correlation_start_date:%Y-%m-%d}")
        model_info.append(f"- Optimal lag period: {self.optimal_lag} months")
        model_info.append("")
        
        # Model comparison
        model_info.append("Model Performance Comparison:")
        model_info.append("-" * 40)
        
        for method, results in self.results_comparison.items():
            if method == 'baseline':
                model_info.append(f"Baseline Linear Model:")
                model_info.append(f"  - Correlation: {results['correlation']:.6f}")
                model_info.append(f"  - Features: {results['n_features']}")
                
            elif method == 'linear':
                model_info.append(f"Linear Model (with feature engineering):")
                model_info.append(f"  - Correlation: {results['correlation']:.6f}")
                model_info.append(f"  - Features: {results['n_features']}")
                model_info.append(f"  - Improvement: +{results['improvement']:.6f}")
                
            elif method == 'best_ml':
                model_info.append(f"Best ML Model ({results['model_name']}):")
                model_info.append(f"  - Test Correlation: {results['test_correlation']:.6f}")
                model_info.append(f"  - Test R²: {results['test_r2']:.6f}")
                model_info.append(f"  - Train R²: {results['train_r2']:.6f}")
        
        model_info.append("")
        model_info.append(f"Final Selected Method: {self.final_method}")
        model_info.append(f"Final Model Correlation: {self.final_correlation:.6f}")
        model_info.append("")
        
        # ML Model Performance Summary if available
        if hasattr(self, 'ml_suite') and self.ml_suite.performance_metrics:
            model_info.append("All ML Model Performance:")
            model_info.append("-" * 30)
            performance_df = self.ml_suite.get_performance_summary()
            for _, row in performance_df.iterrows():
                model_info.append(f"{row['Model']:20} - Test R²: {row['Test_R2']:.4f}, Test Corr: {row['Test_Correlation']:.4f}")
            model_info.append("")
        
        # Feature engineering details
        if hasattr(self, 'feature_engineer'):
            model_info.append("Feature Engineering Applied:")
            model_info.append("- Lagged features (3, 6, 12 months)")
            model_info.append("- Rate of change/momentum (1, 3, 6 months)")
            model_info.append("- Moving averages and ratios (3, 6, 12 months)")
            model_info.append("- Volatility measures (6, 12 months)")
            model_info.append("- Economic interaction terms")
            model_info.append("- Economic ratios (Gold/Silver, Oil/Gold, etc.)")
            model_info.append("- Polynomial features (degree 2)")
            model_info.append("- Seasonal/cyclical features")
            model_info.append("")
        
        # Add replication parameters
        model_info.append("Replication Parameters:")
        model_info.append("-" * 30)
        model_info.append(f"Data Directory: {self.data_dir}")
        model_info.append(f"Model Start Date: {self.model_start_date}")
        model_info.append(f"Correlation Start Date: {self.correlation_start_date}")
        model_info.append(f"Optimal Lag: {self.optimal_lag}")
        model_info.append(f"Use Advanced Features: {self.use_advanced_features}")
        model_info.append(f"Use ML Models: {self.use_ml_models}")
        model_info.append("")
        
        # Add actual model weights for replication
        if self.final_method == 'baseline' and hasattr(self, 'baseline_weights'):
            model_info.append("Baseline Model Weights (for replication):")
            model_info.append("-" * 40)
            for i, col in enumerate(self.baseline_model_data.columns):
                model_info.append(f"{col}: {self.baseline_weights[i]}")
            model_info.append("")
        elif self.final_method == 'linear' and hasattr(self, 'weights'):
            model_info.append("Model Weights (for replication):")
            model_info.append("-" * 40)
            model_info.append("# Format: feature_name: weight_value")
            for i, col in enumerate(self.features.columns):
                model_info.append(f"{col}: {self.weights[i]}")
            model_info.append("")
        
        # Add normalization parameters if available
        if hasattr(self, 'baseline_model_data'):
            model_info.append("Normalization Parameters (for replication):")
            model_info.append("-" * 40)
            model_info.append("# Format: feature_name: mean, std")
            for col in self.baseline_model_data.columns:
                mean = self.baseline_model_data[col].mean()
                std = self.baseline_model_data[col].std()
                model_info.append(f"{col}: mean={mean:.10f}, std={std:.10f}")
            model_info.append("")
        
        model_info.append(f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}")
        
        # Save to file
        output_path = os.path.join(self.output_dir, 'model_params.txt')
        os.makedirs(self.output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(model_info))
            
        print(f"✓ Model parameters saved to {output_path}")
        
    def save_summary_report(self):
        """Save comprehensive summary report"""
        print("\n=== Saving Summary Report ===")
        
        summary = []
        summary.append("============================================================")
        summary.append("TNO FINANCIAL CONDITIONS INDEX - RESULTS SUMMARY")
        summary.append(f"MODEL: {self.final_method.upper()}")
        summary.append("============================================================")
        summary.append("")
        
        # Training information
        summary.append("1. MODEL TRAINING PERIOD")
        if hasattr(self, 'baseline_model_data'):
            summary.append(f"   - Full model data: {self.baseline_model_data.index.min():%Y-%m-%d} to {self.baseline_model_data.index.max():%Y-%m-%d}")
            summary.append(f"   - Records: {len(self.baseline_model_data)}")
        summary.append("")
        
        # ISM correlation information
        summary.append("2. ISM CORRELATION PERIOD")
        if 'ism' in self.monthly_data:
            ism_data = self.monthly_data['ism']['ISM_PMI']
            summary.append(f"   - ISM data available: {ism_data.index.min():%Y-%m-%d} to {ism_data.index.max():%Y-%m-%d}")
            summary.append(f"   - Records with ISM: {len(ism_data)}")
            
            # Calculate actual correlation data used
            ism_forward = ism_data.shift(-self.optimal_lag)
            if hasattr(self, 'correlation_start_date'):
                correlation_data = ism_forward.loc[self.correlation_start_date:].dropna()
                summary.append(f"   - Correlation calculated on: {len(correlation_data)} months")
        summary.append("")
        
        # Model performance
        summary.append("3. MODEL PERFORMANCE")
        summary.append(f"   - Model method: {self.final_method}")
        summary.append(f"   - Correlation with ISM ({self.optimal_lag}-month lead): {self.final_correlation:.6f}")
        summary.append(f"   - R-squared: {self.final_correlation**2:.6f}")
        if hasattr(self, 'baseline_correlation'):
            improvement = self.final_correlation - self.baseline_correlation
            summary.append(f"   - Improvement over baseline: {improvement:+.6f}")
        summary.append("")
        
        # Model weights if linear model
        if self.final_method in ['baseline', 'linear']:
            summary.append("4. OPTIMAL WEIGHTS")
            summary.append(f"   (Using {self.optimal_lag}-month old economic indicators to predict current ISM-PMI)")
            summary.append("")
            
            if self.final_method == 'baseline' and hasattr(self, 'baseline_weights'):
                for i, col in enumerate(self.baseline_model_data.columns):
                    summary.append(f"   {col:15} {self.baseline_weights[i]:7.4f}")
            elif self.final_method == 'linear' and hasattr(self, 'weights'):
                # Show top 20 weights for linear model
                feature_weights = pd.Series(self.weights, index=self.features.columns)
                top_features = feature_weights.abs().sort_values(ascending=False).head(20)
                summary.append("   Top 20 features by absolute weight:")
                for feature, weight in top_features.items():
                    actual_weight = feature_weights[feature]
                    summary.append(f"   {feature[:30]:30} {actual_weight:7.4f}")
        else:
            summary.append("4. ML MODEL WEIGHTS")
            summary.append("   (Weights not applicable for ML models)")
        summary.append("")
        
        # FCI statistics
        summary.append("5. FCI STATISTICS")
        summary.append(f"   - Mean: {self.final_fci.mean():.6f}")
        summary.append(f"   - Std Dev: {self.final_fci.std():.6f}")
        summary.append(f"   - Min: {self.final_fci.min():.6f}")
        summary.append(f"   - Max: {self.final_fci.max():.6f}")
        summary.append("")
        
        # Recent values
        summary.append("6. RECENT VALUES (Last 10 months)")
        summary.append("-" * 60)
        
        # Get recent data
        recent_fci = self.final_fci.tail(10)
        if 'ism' in self.monthly_data:
            recent_ism = self.monthly_data['ism']['ISM_PMI'].tail(10)
            ism_forward = self.monthly_data['ism']['ISM_PMI'].shift(-self.optimal_lag)
            recent_ism_fwd = ism_forward.tail(10)
            
            for date in recent_fci.index:
                fci_val = recent_fci.get(date, 'N/A')
                ism_val = recent_ism.get(date, 'N/A')
                ism_fwd_val = recent_ism_fwd.get(date, 'N/A')
                
                fci_str = f"{fci_val:6.3f}" if fci_val != 'N/A' else "  N/A "
                ism_str = f"{ism_val:4.1f}" if ism_val != 'N/A' else " N/A"
                ism_fwd_str = f"{ism_fwd_val:4.1f}" if ism_fwd_val != 'N/A' else " N/A"
                
                summary.append(f"{date:%Y-%m-%d} | FCI: {fci_str} | ISM: {ism_str}  | ISM {self.optimal_lag}m Fwd: {ism_fwd_str}")
        else:
            for date, fci_val in recent_fci.items():
                summary.append(f"{date:%Y-%m-%d} | FCI: {fci_val:6.3f}")
        summary.append("")
        
        # Interpretation
        summary.append("============================================================")
        summary.append("INTERPRETATION:")
        summary.append(f"- The TNO-FCI uses {self.optimal_lag}-month old economic indicators to predict current ISM-PMI")
        summary.append(f"- Correlation of {self.final_correlation:.4f} indicates {'strong' if self.final_correlation > 0.6 else 'moderate' if self.final_correlation > 0.4 else 'weak'} predictive power")
        
        if len(recent_fci) > 0:
            latest_fci = recent_fci.iloc[-1]
            latest_date = recent_fci.index[-1]
            future_date = latest_date + pd.DateOffset(months=self.optimal_lag)
            
            # Simple linear projection from correlation
            if 'ism' in self.monthly_data:
                ism_mean = self.monthly_data['ism']['ISM_PMI'].mean()
                ism_std = self.monthly_data['ism']['ISM_PMI'].std()
                fci_mean = self.final_fci.mean()
                fci_std = self.final_fci.std()
                
                # Standardized projection
                fci_std_score = (latest_fci - fci_mean) / fci_std
                projected_ism = ism_mean + (fci_std_score * self.final_correlation * ism_std)
                
                summary.append(f"- Current FCI of {latest_fci:.3f} ({latest_date:%b %Y}) projects conditions for {future_date:%b %Y}")
                summary.append(f"- Projected ISM-PMI: {projected_ism:.1f} (suggests {'expansion' if projected_ism > 50 else 'contraction'})")
        
        summary.append("============================================================")
        
        # Save to file
        output_path = os.path.join(self.output_dir, 'fci_summary.txt')
        with open(output_path, 'w') as f:
            f.write('\n'.join(summary))
            
        print(f"✓ Summary report saved to {output_path}")
        
    def archive_results(self):
        """Archive results to timestamped directory"""
        print("\n=== Archiving Results ===")
        
        # Create archive directory
        archive_dir, timestamp = create_archive_directory(output_dir=self.output_dir)
        save_timestamp(timestamp, output_dir=self.output_dir)
        
        print(f"\nArchiving to: {archive_dir}")
        
        # Archive files - only the three main files
        files_to_archive = [
            (os.path.join(self.output_dir, 'fci_results.csv'), 'fci_results.csv'),
            (os.path.join(self.output_dir, 'model_params.txt'), 'model_params.txt'),
            (os.path.join(self.output_dir, 'fci_summary.txt'), 'fci_summary.txt')
        ]
        
        for source_path, archive_name in files_to_archive:
            if os.path.exists(source_path):
                archive_file(source_path, archive_dir, archive_name)
            
        print(f"\n✓ Results archived to {archive_dir}")

    def run(self):
        """Run the complete TNO FCI calculation"""
        print("\n" + "="*70)
        print("    TNO FINANCIAL CONDITIONS INDEX CALCULATOR")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Determine date ranges
        self.determine_date_ranges()
        
        # Prepare monthly data
        self.prepare_monthly_data()
        
        # Optimize lag periods
        self.optimize_lag_periods()
        
        # Build baseline model
        self.build_baseline_model()
        
        # Build advanced model with feature engineering
        if self.use_advanced_features:
            self.build_advanced_model()
        
        # Build ML models
        if self.use_ml_models:
            self.build_ml_models()
        
        # Calculate final FCI using best method
        self.calculate_final_fci()
        
        # For backward compatibility
        self.calculate_fci()
        
        # Save results
        self.save_results()
        
        print("\n✓ TNO FCI calculation complete!")
        
        # Print final summary
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Selected method: {self.final_method}")
        print(f"Final correlation: {self.final_correlation:.6f}")
        print(f"Optimal lag period: {self.optimal_lag} months")
        
        if hasattr(self, 'baseline_correlation'):
            improvement = self.final_correlation - self.baseline_correlation
            print(f"Improvement over baseline: {improvement:+.6f}")
            print(f"R² equivalent improvement: {self.final_correlation**2 - self.baseline_correlation**2:+.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit the TNO-FCI and write the results')
    parser.add_argument('--demo', action='store_true',
                        help='Read demo/data/ and write to demo/output/ (see scripts/make_demo_data.py)')
    args = parser.parse_args()
    root = 'demo' if args.demo else ''
    # Run with all enhancements
    calculator = TNOFCICalculator(data_dir=os.path.join(root, 'data'), use_advanced_features=True,
                                  use_ml_models=True, output_dir=os.path.join(root, 'output'))
    calculator.run()