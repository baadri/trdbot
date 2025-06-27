"""
LSTM Model for Price Prediction
Long Short-Term Memory neural network for time series forecasting
"""

import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Optional, Tuple, Any
import joblib
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM model for cryptocurrency price prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.sequence_length = config.get('sequence_length', 100)
        self.hidden_units = config.get('hidden_units', [256, 128, 64])
        self.dropout = config.get('dropout', 0.2)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        
        # Model architecture parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.validation_split = config.get('validation_split', 0.2)
        
        # Training history
        self.training_history = None
        self.feature_columns = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def prepare_data(self, features_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        
        Args:
            features_dict: Dictionary with multi-timeframe features
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        try:
            # Extract time series data from features
            time_series_data = self._extract_time_series(features_dict)
            
            if time_series_data.empty:
                logger.error("No time series data available")
                return np.array([]), np.array([])
            
            # Create sequences for LSTM
            X, y = self._create_sequences(time_series_data)
            
            if len(X) == 0:
                logger.error("Could not create sequences from data")
                return np.array([]), np.array([])
            
            # Scale the data
            X_scaled = self._scale_features(X)
            y_scaled = self._scale_targets(y)
            
            logger.info(f"Prepared LSTM data: X shape {X_scaled.shape}, y shape {y_scaled.shape}")
            return X_scaled, y_scaled
            
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {e}")
            return np.array([]), np.array([])
    
    def _extract_time_series(self, features_dict: Dict[str, Any]) -> pd.DataFrame:
        """Extract time series data from features dictionary"""
        try:
            # Combine data from all trading pairs and timeframes
            all_data = []
            
            for pair, timeframes_data in features_dict.items():
                for timeframe, timeframe_features in timeframes_data.items():
                    if 'ohlcv' in timeframe_features:
                        ohlcv_data = timeframe_features['ohlcv']
                        
                        if isinstance(ohlcv_data, pd.DataFrame) and not ohlcv_data.empty:
                            # Add technical indicators as additional features
                            enhanced_data = ohlcv_data.copy()
                            
                            # Add technical indicators if available
                            if 'indicators' in timeframe_features:
                                indicators = timeframe_features['indicators']
                                
                                # Add indicator values as new columns
                                for indicator_name, indicator_value in indicators.items():
                                    if isinstance(indicator_value, (int, float)) and not np.isnan(indicator_value):
                                        enhanced_data[f'indicator_{indicator_name}'] = indicator_value
                            
                            # Add pair and timeframe info
                            enhanced_data['pair'] = pair
                            enhanced_data['timeframe'] = timeframe
                            
                            all_data.append(enhanced_data)
            
            if not all_data:
                return pd.DataFrame()
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Sort by timestamp
            if 'timestamp' in combined_data.columns:
                combined_data = combined_data.sort_values('timestamp')
            elif combined_data.index.name == 'timestamp' or isinstance(combined_data.index, pd.DatetimeIndex):
                combined_data = combined_data.sort_index()
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Error extracting time series: {e}")
            return pd.DataFrame()
    
    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        try:
            # Select features for LSTM
            feature_columns = self._select_lstm_features(data)
            self.feature_columns = feature_columns
            
            if not feature_columns:
                logger.error("No suitable features found for LSTM")
                return np.array([]), np.array([])
            
            # Prepare feature matrix
            X_data = data[feature_columns].values
            
            # Create target variable (future price movement)
            y_data = self._create_target_variable(data)
            
            if len(y_data) == 0:
                logger.error("Could not create target variable")
                return np.array([]), np.array([])
            
            # Create sequences
            X_sequences = []
            y_sequences = []
            
            for i in range(self.sequence_length, len(X_data)):
                if i < len(y_data):
                    X_sequences.append(X_data[i-self.sequence_length:i])
                    y_sequences.append(y_data[i])
            
            return np.array(X_sequences), np.array(y_sequences)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])
    
    def _select_lstm_features(self, data: pd.DataFrame) -> List[str]:
        """Select appropriate features for LSTM"""
        feature_columns = []
        
        # OHLCV features
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            if col in data.columns:
                feature_columns.append(col)
        
        # Technical indicator features
        indicator_cols = [col for col in data.columns if col.startswith('indicator_')]
        feature_columns.extend(indicator_cols)
        
        # Additional price-based features
        if 'close' in data.columns:
            # Calculate returns
            data['returns'] = data['close'].pct_change()
            feature_columns.append('returns')
            
            # Calculate moving averages
            for window in [5, 10, 20]:
                ma_col = f'ma_{window}'
                data[ma_col] = data['close'].rolling(window=window).mean()
                if not data[ma_col].isna().all():
                    feature_columns.append(ma_col)
        
        # Remove columns with too many NaN values
        valid_columns = []
        for col in feature_columns:
            if col in data.columns:
                nan_ratio = data[col].isna().sum() / len(data)
                if nan_ratio < 0.5:  # Less than 50% NaN values
                    valid_columns.append(col)
        
        return valid_columns
    
    def _create_target_variable(self, data: pd.DataFrame) -> np.ndarray:
        """Create target variable for prediction"""
        try:
            if 'close' not in data.columns:
                logger.error("Close price not available for target creation")
                return np.array([])
            
            close_prices = data['close'].values
            
            # Predict future price movement (binary classification)
            # 1 = price will go up, 0 = price will go down
            future_steps = 1  # Predict next candle
            targets = []
            
            for i in range(len(close_prices) - future_steps):
                current_price = close_prices[i]
                future_price = close_prices[i + future_steps]
                
                # Binary target: 1 if price goes up, 0 if down
                target = 1 if future_price > current_price else 0
                targets.append(target)
            
            return np.array(targets)
            
        except Exception as e:
            logger.error(f"Error creating target variable: {e}")
            return np.array([])
    
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features using MinMaxScaler"""
        try:
            # Reshape for scaling
            original_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])
            
            # Fit and transform
            X_scaled = self.scaler_X.fit_transform(X_reshaped)
            
            # Reshape back
            X_scaled = X_scaled.reshape(original_shape)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return X
    
    def _scale_targets(self, y: np.ndarray) -> np.ndarray:
        """Scale target values"""
        try:
            # For binary classification, no scaling needed
            return y.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error scaling targets: {e}")
            return y
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
        """
        try:
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(
                self.hidden_units[0],
                return_sequences=True,
                input_shape=input_shape,
                dropout=self.dropout,
                recurrent_dropout=self.dropout
            ))
            model.add(BatchNormalization())
            
            # Additional LSTM layers
            for units in self.hidden_units[1:-1]:
                model.add(LSTM(
                    units,
                    return_sequences=True,
                    dropout=self.dropout,
                    recurrent_dropout=self.dropout
                ))
                model.add(BatchNormalization())
            
            # Final LSTM layer
            model.add(LSTM(
                self.hidden_units[-1],
                return_sequences=False,
                dropout=self.dropout,
                recurrent_dropout=self.dropout
            ))
            model.add(BatchNormalization())
            
            # Dense layers
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(self.dropout))
            
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(self.dropout))
            
            # Output layer (binary classification)
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile model
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.model = model
            
            logger.info(f"LSTM model built with input shape {input_shape}")
            logger.info(f"Model summary: {model.count_params()} parameters")
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            raise
    
    async def train(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the LSTM model
        
        Args:
            features_dict: Dictionary with training features
            
        Returns:
            Training results and metrics
        """
        try:
            logger.info("Starting LSTM model training...")
            
            # Prepare data
            X, y = self.prepare_data(features_dict)
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No training data available")
            
            # Build model
            input_shape = (X.shape[1], X.shape[2])
            self.build_model(input_shape)
            
            # Prepare callbacks
            callbacks = self._prepare_callbacks()
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            self.training_history = history.history
            
            # Evaluate model
            train_metrics = self._evaluate_model(X, y)
            
            logger.info("LSTM model training completed")
            
            return {
                'model_type': 'LSTM',
                'training_samples': len(X),
                'epochs_trained': len(history.history['loss']),
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history['accuracy'][-1],
                'validation_loss': history.history.get('val_loss', [0])[-1],
                'validation_accuracy': history.history.get('val_accuracy', [0])[-1],
                'metrics': train_metrics
            }
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {'error': str(e)}
    
    def _prepare_callbacks(self) -> List:
        """Prepare training callbacks"""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reducer)
        
        # Model checkpoint
        os.makedirs('models/checkpoints', exist_ok=True)
        checkpoint = ModelCheckpoint(
            'models/checkpoints/lstm_best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            # Make predictions
            y_pred = self.model.predict(X)
            y_pred_binary = (y_pred > 0.5).astype(int).flatten()
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y, y_pred_binary)
            precision = precision_score(y, y_pred_binary, average='weighted')
            recall = recall_score(y, y_pred_binary, average='weighted')
            f1 = f1_score(y, y_pred_binary, average='weighted')
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def predict(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make prediction using trained model
        
        Args:
            features: Input features for prediction
            
        Returns:
            Prediction results
        """
        try:
            if self.model is None:
                logger.error("Model not trained yet")
                return None
            
            # Prepare input data
            X, _ = self.prepare_data({f'current': features})
            
            if len(X) == 0:
                logger.warning("No valid input data for prediction")
                return None
            
            # Make prediction
            prediction = self.model.predict(X[-1:])  # Use last sequence
            
            # Convert to probability and direction
            probability = float(prediction[0][0])
            direction = 'up' if probability > 0.5 else 'down'
            confidence = abs(probability - 0.5) * 2  # Scale to 0-1
            
            return {
                'prediction': direction,
                'probability': probability,
                'confidence': confidence,
                'model_type': 'LSTM'
            }
            
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            return None
    
    def save(self, filepath: str) -> None:
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            model_path = filepath.replace('.pkl', '_model.h5')
            if self.model:
                self.model.save(model_path)
            
            # Save scalers and metadata
            model_data = {
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'config': self.config,
                'feature_columns': self.feature_columns,
                'training_history': self.training_history,
                'sequence_length': self.sequence_length
            }
            
            joblib.dump(model_data, filepath)
            
            logger.info(f"LSTM model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving LSTM model: {e}")
    
    def load(self, filepath: str) -> None:
        """Load a trained model"""
        try:
            # Load model data
            model_data = joblib.load(filepath)
            
            self.scaler_X = model_data['scaler_X']
            self.scaler_y = model_data['scaler_y']
            self.config = model_data['config']
            self.feature_columns = model_data['feature_columns']
            self.training_history = model_data['training_history']
            self.sequence_length = model_data['sequence_length']
            
            # Load Keras model
            model_path = filepath.replace('.pkl', '_model.h5')
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
            
            logger.info(f"LSTM model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary and statistics"""
        try:
            summary = {
                'model_type': 'LSTM',
                'architecture': {
                    'sequence_length': self.sequence_length,
                    'hidden_units': self.hidden_units,
                    'dropout': self.dropout,
                    'learning_rate': self.learning_rate
                },
                'training_config': {
                    'epochs': self.epochs,
                    'batch_size': self.batch_size,
                    'validation_split': self.validation_split
                }
            }
            
            if self.model:
                summary['parameters'] = self.model.count_params()
                summary['input_shape'] = self.model.input_shape
                summary['output_shape'] = self.model.output_shape
            
            if self.training_history:
                summary['training_history'] = {
                    'epochs_completed': len(self.training_history['loss']),
                    'best_loss': min(self.training_history['loss']),
                    'best_accuracy': max(self.training_history['accuracy'])
                }
            
            if self.feature_columns:
                summary['features'] = {
                    'count': len(self.feature_columns),
                    'columns': self.feature_columns
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            return {'error': str(e)}