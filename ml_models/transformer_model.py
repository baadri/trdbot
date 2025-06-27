"""
Transformer Model for Trend Analysis
Implements a Transformer architecture for trend prediction
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Embedding
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, Any, Optional
import joblib
import os

logger = logging.getLogger(__name__)

class TransformerModel:
    """Transformer model for trend analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        # Model parameters
        self.context_length = config.get('context_length', 500)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 6)
        self.d_model = config.get('d_model', 512)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 16)
        self.validation_split = config.get('validation_split', 0.2)
        
        # Training history
        self.training_history = None
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def prepare_data(self, features_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for Transformer training
        
        Args:
            features_dict: Dictionary with multi-timeframe features
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        try:
            # Extract and flatten features
            time_series_data = self._extract_time_series(features_dict)
            
            if time_series_data.size == 0:
                logger.error("No valid time series data available for Transformer")
                return np.array([]), np.array([])
            
            # Create sequences
            X, y = self._create_sequences(time_series_data)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("Could not create sequences from time series data")
                return np.array([]), np.array([])
            
            # Scale data
            X_scaled = self._scale_features(X)
            y_scaled = self._scale_targets(y)
            
            return X_scaled, y_scaled
            
        except Exception as e:
            logger.error(f"Error preparing Transformer data: {e}")
            return np.array([]), np.array([])
    
    def _extract_time_series(self, features_dict: Dict[str, Any]) -> np.ndarray:
        """Extract and flatten time series data"""
        try:
            all_features = []
            
            for pair, timeframes_data in features_dict.items():
                for timeframe, timeframe_features in timeframes_data.items():
                    if 'indicators' in timeframe_features:
                        indicator_values = list(timeframe_features['indicators'].values())
                        if len(indicator_values) > 0:
                            all_features.append(indicator_values)
            
            return np.array(all_features)
            
        except Exception as e:
            logger.error(f"Error extracting time series data: {e}")
            return np.array([])
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for Transformer training"""
        try:
            if len(data) < self.context_length:
                logger.error("Insufficient data for sequence creation")
                return np.array([]), np.array([])
            
            X, y = [], []
            
            for i in range(self.context_length, len(data)):
                X.append(data[i-self.context_length:i])
                y.append(data[i][0])  # Predict the first feature as target
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])
    
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale input features"""
        try:
            original_shape = X.shape
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler_X.fit_transform(X_flat)
            return X_scaled.reshape(original_shape)
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return X
    
    def _scale_targets(self, y: np.ndarray) -> np.ndarray:
        """Scale target values"""
        try:
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
            return y_scaled.flatten()
        except Exception as e:
            logger.error(f"Error scaling targets: {e}")
            return y
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build Transformer model architecture
        
        Args:
            input_shape: Shape of input data (context_length, n_features)
        """
        try:
            inputs = Input(shape=input_shape)
            
            # Positional encoding
            x = self._add_positional_encoding(inputs)
            
            # Transformer layers
            for _ in range(self.num_layers):
                x = self._transformer_block(x)
            
            # Global pooling and output
            x = GlobalAveragePooling1D()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
            outputs = Dense(1, activation='linear')(x)
            
            self.model = Model(inputs=inputs, outputs=outputs)
            
            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
            
            logger.info("Transformer model built successfully")
            
        except Exception as e:
            logger.error(f"Error building Transformer model: {e}")
            raise
    
    def _add_positional_encoding(self, inputs: tf.Tensor) -> tf.Tensor:
        """Add positional encoding to inputs"""
        try:
            seq_length = tf.shape(inputs)[1]
            d_model = tf.shape(inputs)[2]
            
            position_indices = tf.range(seq_length, dtype=tf.float32)
            position_indices = tf.expand_dims(position_indices, axis=1)
            
            frequencies = tf.range(d_model, dtype=tf.float32)
            frequencies = tf.expand_dims(frequencies, axis=0)
            
            angles = position_indices / (10000 ** (2 * frequencies / d_model))
            positional_encoding = tf.concat([tf.sin(angles), tf.cos(angles)], axis=1)
            
            inputs = inputs + positional_encoding
            return inputs
            
        except Exception as e:
            logger.error(f"Error adding positional encoding: {e}")
            raise
    
    def _transformer_block(self, x: tf.Tensor) -> tf.Tensor:
        """Single Transformer block"""
        try:
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.d_model
            )(x, x)
            
            # Add & Norm
            x = x + attention_output
            x = LayerNormalization(epsilon=1e-6)(x)
            
            # Feed-forward network
            ff_output = Dense(self.d_model * 4, activation='relu')(x)
            ff_output = Dense(self.d_model)(ff_output)
            
            # Add & Norm
            x = x + ff_output
            x = LayerNormalization(epsilon=1e-6)(x)
            
            return x
            
        except Exception as e:
            logger.error(f"Error in Transformer block: {e}")
            raise
    
    async def train(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the Transformer model
        
        Args:
            features_dict: Dictionary with training features
            
        Returns:
            Training results
        """
        try:
            logger.info("Starting Transformer model training...")
            
            # Prepare data
            X, y = self.prepare_data(features_dict)
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No training data available")
            
            # Build model
            input_shape = (X.shape[1], X.shape[2])
            self.build_model(input_shape)
            
            # Callbacks
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
            
            logger.info("Transformer model training completed")
            
            return {
                'model_type': 'Transformer',
                'training_samples': len(X),
                'epochs_trained': len(history.history['loss']),
                'final_loss': history.history['loss'][-1],
                'final_mae': history.history['mae'][-1]
            }
            
        except Exception as e:
            logger.error(f"Error training Transformer model: {e}")
            return {'error': str(e)}
    
    def _prepare_callbacks(self) -> list:
        """Prepare training callbacks"""
        callbacks = []
        
        # Early stopping
        callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))
        
        # Reduce learning rate on plateau
        callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7))
        
        # Model checkpoint
        os.makedirs('models/checkpoints', exist_ok=True)
        callbacks.append(ModelCheckpoint('models/checkpoints/transformer_best_model.h5', save_best_only=True))
        
        return callbacks
    
    def predict(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make prediction using the trained Transformer model
        
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
            X, _ = self.prepare_data({'current': features})
            
            if len(X) == 0:
                logger.warning("No valid input data for prediction")
                return None
            
            # Make prediction
            prediction = self.model.predict(X[-1:])
            return {'prediction': float(prediction[0]), 'model_type': 'Transformer'}
            
        except Exception as e:
            logger.error(f"Error making Transformer prediction: {e}")
            return None
    
    def save(self, filepath: str) -> None:
        """Save the trained Transformer model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            model_path = filepath.replace('.pkl', '_model.h5')
            if self.model:
                self.model.save(model_path)
            
            # Save scalers and config
            model_data = {
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'config': self.config,
                'training_history': self.training_history
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Transformer model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving Transformer model: {e}")
    
    def load(self, filepath: str) -> None:
        """Load a trained Transformer model"""
        try:
            model_data = joblib.load(filepath)
            self.scaler_X = model_data['scaler_X']
            self.scaler_y = model_data['scaler_y']
            self.config = model_data['config']
            self.training_history = model_data['training_history']
            
            model_path = filepath.replace('.pkl', '_model.h5')
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
            
            logger.info(f"Transformer model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading Transformer model: {e}")