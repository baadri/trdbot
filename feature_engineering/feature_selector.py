"""
Feature Selector
Selects and ranks the most important features for ML models
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FeatureSelector:
    """Advanced feature selection for trading signals"""
    
    def __init__(self):
        self.selected_features = {}
        self.feature_importance = {}
        self.correlation_matrix = None
        self.scaler = StandardScaler()
        
        # Feature selection methods
        self.selection_methods = {
            'univariate': self._univariate_selection,
            'mutual_info': self._mutual_info_selection,
            'correlation': self._correlation_selection,
            'rfe': self._recursive_feature_elimination,
            'lasso': self._lasso_selection,
            'random_forest': self._random_forest_selection
        }
    
    def select_features(self, features_dict: Dict[str, Any], target_variable: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Select best features from multi-timeframe feature dictionary
        
        Args:
            features_dict: Dictionary with timeframe features
            target_variable: Target variable for supervised selection
            
        Returns:
            Dictionary with selected features
        """
        try:
            # Flatten features from all timeframes
            flattened_features = self._flatten_features(features_dict)
            
            if not flattened_features:
                logger.warning("No features to select from")
                return {}
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([flattened_features])
            
            # Remove non-numeric features
            numeric_features = self._get_numeric_features(feature_df)
            
            if numeric_features.empty:
                logger.warning("No numeric features found")
                return {}
            
            # Apply feature selection methods
            selected_features = self._apply_selection_methods(
                numeric_features, target_variable
            )
            
            # Rank and combine results
            final_features = self._rank_and_combine_features(selected_features)
            
            # Create selected feature dictionary
            result = self._create_selected_feature_dict(final_features, flattened_features)
            
            logger.debug(f"Selected {len(result)} features from {len(flattened_features)}")
            return result
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return flattened_features if 'flattened_features' in locals() else {}
    
    def _flatten_features(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested feature dictionary"""
        flattened = {}
        
        try:
            for timeframe, timeframe_data in features_dict.items():
                if isinstance(timeframe_data, dict):
                    for category, category_data in timeframe_data.items():
                        if category == 'indicators' and isinstance(category_data, dict):
                            # Technical indicators
                            for indicator, value in category_data.items():
                                if self._is_valid_feature_value(value):
                                    flattened[f"{timeframe}_{indicator}"] = value
                        
                        elif category == 'microstructure' and isinstance(category_data, dict):
                            # Market microstructure features
                            for feature, value in category_data.items():
                                if self._is_valid_feature_value(value):
                                    flattened[f"{timeframe}_micro_{feature}"] = value
                        
                        elif category == 'ohlcv' and isinstance(category_data, pd.DataFrame):
                            # OHLCV-derived features
                            if not category_data.empty:
                                # Price-based features
                                close_price = category_data['close'].iloc[-1]
                                volume = category_data['volume'].iloc[-1]
                                
                                flattened[f"{timeframe}_close_price"] = close_price
                                flattened[f"{timeframe}_volume"] = volume
                                
                                # Price change features
                                if len(category_data) >= 2:
                                    price_change = (close_price - category_data['close'].iloc[-2]) / category_data['close'].iloc[-2]
                                    flattened[f"{timeframe}_price_change"] = price_change
                                
                                # Volatility features
                                if len(category_data) >= 20:
                                    returns = category_data['close'].pct_change().dropna()
                                    volatility = returns.std()
                                    flattened[f"{timeframe}_volatility"] = volatility
        
        except Exception as e:
            logger.error(f"Error flattening features: {e}")
        
        return flattened
    
    def _is_valid_feature_value(self, value: Any) -> bool:
        """Check if a value is valid for feature selection"""
        if value is None:
            return False
        
        if isinstance(value, (int, float)):
            return not (np.isnan(value) or np.isinf(value))
        
        if isinstance(value, bool):
            return True
        
        if isinstance(value, str):
            # Convert categorical strings to numeric if possible
            return value.lower() in ['true', 'false', 'bullish', 'bearish', 'neutral', 'up', 'down']
        
        return False
    
    def _get_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and convert features to numeric format"""
        numeric_df = df.copy()
        
        try:
            for column in df.columns:
                if df[column].dtype == 'object':
                    # Convert categorical features to numeric
                    if df[column].iloc[0] in ['true', 'false', True, False]:
                        numeric_df[column] = df[column].astype(bool).astype(int)
                    elif df[column].iloc[0] in ['bullish', 'bearish', 'neutral']:
                        mapping = {'bullish': 1, 'neutral': 0, 'bearish': -1}
                        numeric_df[column] = df[column].map(mapping)
                    elif df[column].iloc[0] in ['up', 'down']:
                        mapping = {'up': 1, 'down': -1}
                        numeric_df[column] = df[column].map(mapping)
                    else:
                        # Drop non-convertible categorical features
                        numeric_df = numeric_df.drop(columns=[column])
                elif df[column].dtype in ['int64', 'float64']:
                    # Check for infinite or NaN values
                    if np.any(np.isnan(df[column])) or np.any(np.isinf(df[column])):
                        numeric_df[column] = df[column].fillna(0).replace([np.inf, -np.inf], 0)
            
            return numeric_df
            
        except Exception as e:
            logger.error(f"Error converting to numeric features: {e}")
            return df
    
    def _apply_selection_methods(self, feature_df: pd.DataFrame, target: Optional[np.ndarray]) -> Dict[str, List[str]]:
        """Apply multiple feature selection methods"""
        selected_features = {}
        
        try:
            # Always apply correlation-based selection
            selected_features['correlation'] = self._correlation_selection(feature_df)
            
            # Apply supervised methods if target is available
            if target is not None and len(target) == len(feature_df):
                try:
                    selected_features['univariate'] = self._univariate_selection(feature_df, target)
                    selected_features['mutual_info'] = self._mutual_info_selection(feature_df, target)
                    selected_features['rfe'] = self._recursive_feature_elimination(feature_df, target)
                    selected_features['lasso'] = self._lasso_selection(feature_df, target)
                    selected_features['random_forest'] = self._random_forest_selection(feature_df, target)
                except Exception as e:
                    logger.warning(f"Error in supervised feature selection: {e}")
            
            # Apply unsupervised methods
            try:
                selected_features['variance'] = self._variance_selection(feature_df)
                selected_features['pca_importance'] = self._pca_importance_selection(feature_df)
            except Exception as e:
                logger.warning(f"Error in unsupervised feature selection: {e}")
            
        except Exception as e:
            logger.error(f"Error applying selection methods: {e}")
        
        return selected_features
    
    def _univariate_selection(self, X: pd.DataFrame, y: np.ndarray = None, k: int = 20) -> List[str]:
        """Univariate feature selection using f_classif"""
        try:
            if y is None:
                return list(X.columns[:k])
            
            # Convert target to binary classification if needed
            y_binary = self._convert_to_binary_target(y)
            
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y_binary)
            
            selected_features = X.columns[selector.get_support()].tolist()
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in univariate selection: {e}")
            return list(X.columns[:k])
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: np.ndarray = None, k: int = 20) -> List[str]:
        """Mutual information feature selection"""
        try:
            if y is None:
                return list(X.columns[:k])
            
            y_binary = self._convert_to_binary_target(y)
            
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y_binary)
            
            selected_features = X.columns[selector.get_support()].tolist()
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in mutual info selection: {e}")
            return list(X.columns[:k])
    
    def _correlation_selection(self, X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """Remove highly correlated features"""
        try:
            corr_matrix = X.corr().abs()
            self.correlation_matrix = corr_matrix
            
            # Find pairs of highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > threshold:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for feature1, feature2 in high_corr_pairs:
                # Remove the feature with lower variance
                if X[feature1].var() < X[feature2].var():
                    features_to_remove.add(feature1)
                else:
                    features_to_remove.add(feature2)
            
            selected_features = [col for col in X.columns if col not in features_to_remove]
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in correlation selection: {e}")
            return list(X.columns)
    
    def _recursive_feature_elimination(self, X: pd.DataFrame, y: np.ndarray = None, n_features: int = 15) -> List[str]:
        """Recursive Feature Elimination"""
        try:
            if y is None:
                return list(X.columns[:n_features])
            
            y_binary = self._convert_to_binary_target(y)
            
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(n_features, X.shape[1]))
            
            X_selected = selector.fit_transform(X, y_binary)
            selected_features = X.columns[selector.get_support()].tolist()
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in RFE selection: {e}")
            return list(X.columns[:n_features])
    
    def _lasso_selection(self, X: pd.DataFrame, y: np.ndarray = None, alpha: float = 0.01) -> List[str]:
        """LASSO feature selection"""
        try:
            if y is None:
                return list(X.columns)
            
            # Scale features for LASSO
            X_scaled = self.scaler.fit_transform(X)
            
            # Convert to regression target
            y_continuous = self._convert_to_continuous_target(y)
            
            lasso = LassoCV(cv=3, random_state=42)
            lasso.fit(X_scaled, y_continuous)
            
            # Select features with non-zero coefficients
            selected_mask = np.abs(lasso.coef_) > alpha
            selected_features = X.columns[selected_mask].tolist()
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in LASSO selection: {e}")
            return list(X.columns)
    
    def _random_forest_selection(self, X: pd.DataFrame, y: np.ndarray = None, n_features: int = 20) -> List[str]:
        """Random Forest feature importance selection"""
        try:
            if y is None:
                return list(X.columns[:n_features])
            
            y_binary = self._convert_to_binary_target(y)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y_binary)
            
            # Get feature importances
            importances = rf.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Store feature importance
            self.feature_importance['random_forest'] = feature_importance_df
            
            selected_features = feature_importance_df.head(n_features)['feature'].tolist()
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in Random Forest selection: {e}")
            return list(X.columns[:n_features])
    
    def _variance_selection(self, X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """Remove low variance features"""
        try:
            # Calculate variance for each feature
            variances = X.var()
            
            # Select features with variance above threshold
            selected_features = variances[variances > threshold].index.tolist()
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in variance selection: {e}")
            return list(X.columns)
    
    def _pca_importance_selection(self, X: pd.DataFrame, n_components: int = 20) -> List[str]:
        """Select features based on PCA component importance"""
        try:
            from sklearn.decomposition import PCA
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Apply PCA
            pca = PCA(n_components=min(n_components, X.shape[1]))
            pca.fit(X_scaled)
            
            # Calculate feature importance based on PCA components
            feature_importance = np.sum(np.abs(pca.components_), axis=0)
            
            # Select top features
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            selected_features = importance_df.head(n_components)['feature'].tolist()
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in PCA importance selection: {e}")
            return list(X.columns[:n_components])
    
    def _convert_to_binary_target(self, y: np.ndarray) -> np.ndarray:
        """Convert target variable to binary classification"""
        try:
            if len(np.unique(y)) == 2:
                return y
            
            # Convert to binary based on median
            median = np.median(y)
            return (y > median).astype(int)
            
        except Exception:
            return np.ones(len(y), dtype=int)
    
    def _convert_to_continuous_target(self, y: np.ndarray) -> np.ndarray:
        """Convert target variable to continuous values"""
        try:
            return y.astype(float)
        except Exception:
            return np.zeros(len(y), dtype=float)
    
    def _rank_and_combine_features(self, selected_features: Dict[str, List[str]]) -> List[str]:
        """Rank and combine features from different selection methods"""
        try:
            # Count how many times each feature was selected
            feature_counts = {}
            
            for method, features in selected_features.items():
                for feature in features:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            # Sort features by selection count
            ranked_features = sorted(
                feature_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Return top features (selected by at least 2 methods or top 30)
            final_features = []
            for feature, count in ranked_features:
                if count >= 2 or len(final_features) < 30:
                    final_features.append(feature)
                
                if len(final_features) >= 50:  # Limit total features
                    break
            
            return final_features
            
        except Exception as e:
            logger.error(f"Error ranking features: {e}")
            # Fallback: return features from first method
            if selected_features:
                return list(selected_features.values())[0]
            return []
    
    def _create_selected_feature_dict(self, selected_features: List[str], original_features: Dict[str, Any]) -> Dict[str, Any]:
        """Create dictionary with only selected features"""
        try:
            selected_dict = {}
            
            for feature_name in selected_features:
                if feature_name in original_features:
                    selected_dict[feature_name] = original_features[feature_name]
            
            return selected_dict
            
        except Exception as e:
            logger.error(f"Error creating selected feature dict: {e}")
            return original_features
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get summary of feature importance from different methods"""
        try:
            summary = {}
            
            if 'random_forest' in self.feature_importance:
                rf_importance = self.feature_importance['random_forest']
                summary['top_rf_features'] = rf_importance.head(10).to_dict('records')
            
            if self.correlation_matrix is not None:
                # Find most correlated feature pairs
                corr_pairs = []
                for i in range(len(self.correlation_matrix.columns)):
                    for j in range(i+1, len(self.correlation_matrix.columns)):
                        corr_val = self.correlation_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            corr_pairs.append({
                                'feature1': self.correlation_matrix.columns[i],
                                'feature2': self.correlation_matrix.columns[j],
                                'correlation': corr_val
                            })
                
                summary['high_correlation_pairs'] = sorted(
                    corr_pairs, key=lambda x: abs(x['correlation']), reverse=True
                )[:10]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating feature importance summary: {e}")
            return {}
    
    def analyze_feature_stability(self, feature_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stability of features over time"""
        try:
            if not feature_history:
                return {}
            
            # Track which features appear consistently
            all_features = set()
            for features in feature_history:
                all_features.update(features.keys())
            
            feature_stability = {}
            for feature in all_features:
                appearances = sum(1 for features in feature_history if feature in features)
                stability_score = appearances / len(feature_history)
                feature_stability[feature] = stability_score
            
            # Find most stable features
            stable_features = [
                feature for feature, stability in feature_stability.items()
                if stability >= 0.8
            ]
            
            return {
                'stable_features': stable_features,
                'stability_scores': feature_stability,
                'total_features_analyzed': len(all_features),
                'avg_stability': np.mean(list(feature_stability.values()))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing feature stability: {e}")
            return {}