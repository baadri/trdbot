"""
XGBoost Model for Volume Analysis
Gradient Boosted Decision Trees for predicting price movements based on volume and other features
"""

import logging
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class XGBoostVolumeModel:
    """XGBoost model for volume-based trading signal generation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None

        # Model hyperparameters
        self.n_estimators = config.get('n_estimators', 100)
        self.max_depth = config.get('max_depth', 6)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.subsample = config.get('subsample', 0.8)
        self.colsample_bytree = config.get('colsample_bytree', 0.8)
        self.objective = config.get('objective', 'binary:logistic')

    def prepare_data(self, features_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for XGBoost training

        Args:
            features_dict: Dictionary with training features

        Returns:
            Tuple of (X, y) arrays
        """
        try:
            data = []
            labels = []

            for pair, timeframes_data in features_dict.items():
                for timeframe, features in timeframes_data.items():
                    if 'indicators' in features and 'ohlcv' in features:
                        indicators = features['indicators']
                        if 'volume' in features['ohlcv'].columns:
                            volume = features['ohlcv']['volume'].iloc[-1]
                            indicators['volume'] = volume
                        data.append(list(indicators.values()))
                        labels.append(1 if indicators.get('target', 0) > 0 else 0)

            if not data or not labels:
                logger.error("No data available for XGBoost training")
                return np.array([]), np.array([])

            self.feature_columns = [key for key in indicators.keys()]
            X = np.array(data)
            y = np.array(labels)

            # Scale data
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            return X_scaled, y

        except Exception as e:
            logger.error(f"Error preparing XGBoost data: {e}")
            return np.array([]), np.array([])

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost model"""
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                objective=self.objective,
                use_label_encoder=False,
                eval_metric="logloss"
            )

            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

            logger.info("XGBoost model training completed")
            self._log_feature_importance()

        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict trading signal using the trained model

        Args:
            features: Dictionary with input features

        Returns:
            Dictionary with prediction results
        """
        try:
            if self.model is None:
                logger.error("Model not trained yet")
                return {}

            data = [features.get(key, 0) for key in self.feature_columns]
            X = np.array([data])
            X_scaled = self.scaler.transform(X)

            prediction = self.model.predict(X_scaled)
            confidence = max(self.model.predict_proba(X_scaled)[0])

            return {
                "signal": "buy" if prediction[0] == 1 else "sell",
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error making prediction with XGBoost model: {e}")
            return {}

    def save(self, filepath: str) -> None:
        """Save the trained model"""
        try:
            joblib.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns
            }, filepath)
            logger.info(f"XGBoost model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving XGBoost model: {e}")

    def load(self, filepath: str) -> None:
        """Load a trained model"""
        try:
            data = joblib.load(filepath)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_columns = data["feature_columns"]
            logger.info(f"XGBoost model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {e}")

    def _log_feature_importance(self):
        """Log feature importance from the trained model"""
        try:
            importance = self.model.get_booster().get_score(importance_type="weight")
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            logger.info("Feature importance:")
            for feature, score in sorted_importance:
                logger.info(f"{feature}: {score}")
        except Exception as e:
            logger.error(f"Error logging feature importance: {e}")