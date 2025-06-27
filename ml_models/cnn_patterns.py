"""
CNN Pattern Recognition Model
Convolutional Neural Network for identifying chart patterns
"""

import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten, 
    BatchNormalization, Input, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Optional, Tuple, Any
import cv2
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class CNNPatternModel:
    """CNN model for chart pattern recognition"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.label_encoder = LabelEncoder()
        
        # Image parameters
        self.image_size = config.get('image_size', [64, 64])
        self.filters = config.get('filters', [32, 64, 128])
        self.kernel_size = config.get('kernel_size', 3)
        
        # Training parameters
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.validation_split = config.get('validation_split', 0.2)
        
        # Pattern classes
        self.pattern_classes = [
            'head_and_shoulders', 'inverse_head_and_shoulders',
            'double_top', 'double_bottom',
            'ascending_triangle', 'descending_triangle', 'symmetrical_triangle',
            'rising_wedge', 'falling_wedge',
            'bullish_flag', 'bearish_flag',
            'bullish_pennant', 'bearish_pennant',
            'cup_and_handle',
            'support_bounce', 'resistance_rejection',
            'breakout_up', 'breakout_down',
            'consolidation', 'trend_continuation',
            'reversal_up', 'reversal_down',
            'no_pattern'
        ]
        
        # Training history
        self.training_history = None
        
        # Set random seeds
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def prepare_data(self, features_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for CNN training by converting price data to images
        
        Args:
            features_dict: Dictionary with multi-timeframe features
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        try:
            images = []
            labels = []
            
            # Process each trading pair and timeframe
            for pair, timeframes_data in features_dict.items():
                for timeframe, timeframe_features in timeframes_data.items():
                    if 'ohlcv' in timeframe_features:
                        ohlcv_data = timeframe_features['ohlcv']
                        
                        if isinstance(ohlcv_data, pd.DataFrame) and len(ohlcv_data) >= 50:
                            # Create multiple overlapping windows
                            window_size = 50
                            step_size = 10
                            
                            for i in range(0, len(ohlcv_data) - window_size, step_size):
                                window_data = ohlcv_data.iloc[i:i+window_size]
                                
                                # Convert to image
                                image = self._create_chart_image(window_data)
                                if image is not None:
                                    images.append(image)
                                    
                                    # Detect pattern and assign label
                                    pattern = self._detect_pattern(window_data)
                                    labels.append(pattern)
            
            if not images:
                logger.error("No images created from data")
                return np.array([]), np.array([])
            
            # Convert to numpy arrays
            X = np.array(images)
            y = np.array(labels)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Normalize images
            X = X.astype('float32') / 255.0
            
            logger.info(f"Prepared CNN data: {len(X)} images, {len(np.unique(y))} pattern classes")
            return X, y_encoded
            
        except Exception as e:
            logger.error(f"Error preparing CNN data: {e}")
            return np.array([]), np.array([])
    
    def _create_chart_image(self, ohlcv_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Convert OHLCV data to candlestick chart image"""
        try:
            if len(ohlcv_data) < 10:
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(6.4, 6.4))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            
            # Prepare data
            opens = ohlcv_data['open'].values
            highs = ohlcv_data['high'].values
            lows = ohlcv_data['low'].values
            closes = ohlcv_data['close'].values
            
            # Normalize prices to 0-1 range for better visualization
            all_prices = np.concatenate([opens, highs, lows, closes])
            price_min, price_max = all_prices.min(), all_prices.max()
            price_range = price_max - price_min
            
            if price_range == 0:
                return None
            
            # Draw candlesticks
            for i in range(len(ohlcv_data)):
                open_price = (opens[i] - price_min) / price_range
                high_price = (highs[i] - price_min) / price_range
                low_price = (lows[i] - price_min) / price_range
                close_price = (closes[i] - price_min) / price_range
                
                # Color based on price movement
                color = 'green' if close_price >= open_price else 'red'
                
                # Draw high-low line
                ax.plot([i, i], [low_price, high_price], color='white', linewidth=1)
                
                # Draw open-close rectangle
                height = abs(close_price - open_price)
                bottom = min(open_price, close_price)
                
                rect = plt.Rectangle((i-0.4, bottom), 0.8, height, 
                                   facecolor=color, edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
            
            # Format chart
            ax.set_xlim(-1, len(ohlcv_data))
            ax.set_ylim(-0.1, 1.1)
            ax.axis('off')
            
            # Convert to image array
            fig.canvas.draw()
            image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            # Resize to target size
            image_resized = cv2.resize(image_array, tuple(self.image_size))
            
            return image_resized
            
        except Exception as e:
            logger.error(f"Error creating chart image: {e}")
            return None
    
    def _detect_pattern(self, ohlcv_data: pd.DataFrame) -> str:
        """Detect chart patterns in OHLCV data"""
        try:
            if len(ohlcv_data) < 20:
                return 'no_pattern'
            
            highs = ohlcv_data['high'].values
            lows = ohlcv_data['low'].values
            closes = ohlcv_data['close'].values
            
            # Simple pattern detection logic
            # In a production system, this would be much more sophisticated
            
            # Trend analysis
            start_price = closes[0]
            end_price = closes[-1]
            mid_price = closes[len(closes)//2]
            
            price_change = (end_price - start_price) / start_price
            
            # Detect basic patterns
            if self._is_double_top(highs, closes):
                return 'double_top'
            elif self._is_double_bottom(lows, closes):
                return 'double_bottom'
            elif self._is_head_and_shoulders(highs, closes):
                return 'head_and_shoulders'
            elif self._is_inverse_head_and_shoulders(lows, closes):
                return 'inverse_head_and_shoulders'
            elif self._is_ascending_triangle(highs, lows):
                return 'ascending_triangle'
            elif self._is_descending_triangle(highs, lows):
                return 'descending_triangle'
            elif self._is_bullish_flag(closes):
                return 'bullish_flag'
            elif self._is_bearish_flag(closes):
                return 'bearish_flag'
            elif abs(price_change) < 0.02:  # Less than 2% change
                return 'consolidation'
            elif price_change > 0.05:  # More than 5% up
                return 'breakout_up'
            elif price_change < -0.05:  # More than 5% down
                return 'breakout_down'
            else:
                return 'no_pattern'
                
        except Exception as e:
            logger.error(f"Error detecting pattern: {e}")
            return 'no_pattern'
    
    def _is_double_top(self, highs: np.ndarray, closes: np.ndarray) -> bool:
        """Detect double top pattern"""
        try:
            if len(highs) < 20:
                return False
            
            # Find peaks
            peaks = []
            for i in range(2, len(highs)-2):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1] and \
                   highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                    peaks.append((i, highs[i]))
            
            if len(peaks) >= 2:
                # Check if two highest peaks are similar
                peaks.sort(key=lambda x: x[1], reverse=True)
                peak1, peak2 = peaks[0], peaks[1]
                
                # Similar heights and sufficient separation
                height_diff = abs(peak1[1] - peak2[1]) / max(peak1[1], peak2[1])
                time_separation = abs(peak1[0] - peak2[0])
                
                if height_diff < 0.03 and time_separation > 5:  # 3% height difference, 5 candles apart
                    # Check if price declined after second peak
                    second_peak_idx = max(peak1[0], peak2[0])
                    if second_peak_idx < len(closes) - 3:
                        price_decline = (closes[-1] - closes[second_peak_idx]) / closes[second_peak_idx]
                        return price_decline < -0.02  # 2% decline
            
            return False
            
        except Exception:
            return False
    
    def _is_double_bottom(self, lows: np.ndarray, closes: np.ndarray) -> bool:
        """Detect double bottom pattern"""
        try:
            if len(lows) < 20:
                return False
            
            # Find troughs
            troughs = []
            for i in range(2, len(lows)-2):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1] and \
                   lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                    troughs.append((i, lows[i]))
            
            if len(troughs) >= 2:
                # Check if two lowest troughs are similar
                troughs.sort(key=lambda x: x[1])
                trough1, trough2 = troughs[0], troughs[1]
                
                height_diff = abs(trough1[1] - trough2[1]) / min(trough1[1], trough2[1])
                time_separation = abs(trough1[0] - trough2[0])
                
                if height_diff < 0.03 and time_separation > 5:
                    # Check if price increased after second trough
                    second_trough_idx = max(trough1[0], trough2[0])
                    if second_trough_idx < len(closes) - 3:
                        price_increase = (closes[-1] - closes[second_trough_idx]) / closes[second_trough_idx]
                        return price_increase > 0.02  # 2% increase
            
            return False
            
        except Exception:
            return False
    
    def _is_head_and_shoulders(self, highs: np.ndarray, closes: np.ndarray) -> bool:
        """Detect head and shoulders pattern"""
        try:
            if len(highs) < 15:
                return False
            
            # Find three peaks
            peaks = []
            for i in range(2, len(highs)-2):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
            
            if len(peaks) >= 3:
                # Sort by height
                peaks.sort(key=lambda x: x[1], reverse=True)
                
                # Check for head and shoulders pattern
                head = peaks[0]
                shoulders = peaks[1:3]
                
                # Head should be higher than shoulders
                shoulder_heights = [s[1] for s in shoulders]
                avg_shoulder_height = np.mean(shoulder_heights)
                
                if head[1] > avg_shoulder_height * 1.02:  # Head 2% higher
                    # Check shoulder symmetry
                    shoulder_diff = abs(shoulders[0][1] - shoulders[1][1]) / avg_shoulder_height
                    if shoulder_diff < 0.05:  # Shoulders within 5% of each other
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _is_inverse_head_and_shoulders(self, lows: np.ndarray, closes: np.ndarray) -> bool:
        """Detect inverse head and shoulders pattern"""
        try:
            if len(lows) < 15:
                return False
            
            # Find three troughs
            troughs = []
            for i in range(2, len(lows)-2):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    troughs.append((i, lows[i]))
            
            if len(troughs) >= 3:
                # Sort by depth (lowest first)
                troughs.sort(key=lambda x: x[1])
                
                head = troughs[0]  # Deepest trough
                shoulders = troughs[1:3]
                
                # Head should be lower than shoulders
                shoulder_lows = [s[1] for s in shoulders]
                avg_shoulder_low = np.mean(shoulder_lows)
                
                if head[1] < avg_shoulder_low * 0.98:  # Head 2% lower
                    # Check shoulder symmetry
                    shoulder_diff = abs(shoulders[0][1] - shoulders[1][1]) / avg_shoulder_low
                    if shoulder_diff < 0.05:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _is_ascending_triangle(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Detect ascending triangle pattern"""
        try:
            if len(highs) < 15:
                return False
            
            # Check if highs are relatively flat (resistance)
            high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
            
            # Check if lows are ascending (support)
            low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
            
            # Ascending triangle: flat resistance, ascending support
            return abs(high_trend) < 0.001 and low_trend > 0.001
            
        except Exception:
            return False
    
    def _is_descending_triangle(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Detect descending triangle pattern"""
        try:
            if len(highs) < 15:
                return False
            
            # Check if lows are relatively flat (support)
            low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
            
            # Check if highs are descending (resistance)
            high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
            
            # Descending triangle: descending resistance, flat support
            return abs(low_trend) < 0.001 and high_trend < -0.001
            
        except Exception:
            return False
    
    def _is_bullish_flag(self, closes: np.ndarray) -> bool:
        """Detect bullish flag pattern"""
        try:
            if len(closes) < 10:
                return False
            
            # Check for initial uptrend followed by consolidation
            first_half = closes[:len(closes)//2]
            second_half = closes[len(closes)//2:]
            
            # First half should be trending up
            first_trend = np.polyfit(range(len(first_half)), first_half, 1)[0]
            
            # Second half should be consolidating (small downward drift)
            second_trend = np.polyfit(range(len(second_half)), second_half, 1)[0]
            
            return first_trend > 0.002 and -0.001 < second_trend < 0.001
            
        except Exception:
            return False
    
    def _is_bearish_flag(self, closes: np.ndarray) -> bool:
        """Detect bearish flag pattern"""
        try:
            if len(closes) < 10:
                return False
            
            # Check for initial downtrend followed by consolidation
            first_half = closes[:len(closes)//2]
            second_half = closes[len(closes)//2:]
            
            # First half should be trending down
            first_trend = np.polyfit(range(len(first_half)), first_half, 1)[0]
            
            # Second half should be consolidating (small upward drift)
            second_trend = np.polyfit(range(len(second_half)), second_half, 1)[0]
            
            return first_trend < -0.002 and -0.001 < second_trend < 0.001
            
        except Exception:
            return False
    
    def build_model(self, input_shape: Tuple[int, int, int], num_classes: int) -> None:
        """
        Build CNN model architecture
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of pattern classes
        """
        try:
            model = Sequential()
            
            # First convolutional block
            model.add(Conv2D(self.filters[0], (self.kernel_size, self.kernel_size), 
                           activation='relu', input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.25))
            
            # Second convolutional block
            model.add(Conv2D(self.filters[1], (self.kernel_size, self.kernel_size), 
                           activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.25))
            
            # Third convolutional block
            model.add(Conv2D(self.filters[2], (self.kernel_size, self.kernel_size), 
                           activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.25))
            
            # Global average pooling
            model.add(GlobalAveragePooling2D())
            
            # Dense layers
            model.add(Dense(512, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            
            model.add(Dense(256, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            
            # Output layer
            model.add(Dense(num_classes, activation='softmax'))
            
            # Compile model
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            
            logger.info(f"CNN model built with input shape {input_shape}, {num_classes} classes")
            logger.info(f"Model parameters: {model.count_params()}")
            
        except Exception as e:
            logger.error(f"Error building CNN model: {e}")
            raise
    
    async def train(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the CNN model
        
        Args:
            features_dict: Dictionary with training features
            
        Returns:
            Training results and metrics
        """
        try:
            logger.info("Starting CNN pattern recognition model training...")
            
            # Prepare data
            X, y = self.prepare_data(features_dict)
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No training data available")
            
            # Build model
            input_shape = X.shape[1:]  # (height, width, channels)
            num_classes = len(np.unique(y))
            self.build_model(input_shape, num_classes)
            
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
            
            logger.info("CNN pattern recognition model training completed")
            
            return {
                'model_type': 'CNN_Pattern',
                'training_samples': len(X),
                'pattern_classes': num_classes,
                'epochs_trained': len(history.history['loss']),
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history['accuracy'][-1],
                'validation_loss': history.history.get('val_loss', [0])[-1],
                'validation_accuracy': history.history.get('val_accuracy', [0])[-1],
                'metrics': train_metrics
            }
            
        except Exception as e:
            logger.error(f"Error training CNN model: {e}")
            return {'error': str(e)}
    
    def _prepare_callbacks(self) -> List:
        """Prepare training callbacks"""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reducer)
        
        # Model checkpoint
        os.makedirs('models/checkpoints', exist_ok=True)
        checkpoint = ModelCheckpoint(
            'models/checkpoints/cnn_best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            # Make predictions
            y_pred = self.model.predict(X)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y, y_pred_classes)
            precision = precision_score(y, y_pred_classes, average='weighted')
            recall = recall_score(y, y_pred_classes, average='weighted')
            f1 = f1_score(y, y_pred_classes, average='weighted')
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating CNN model: {e}")
            return {}
    
    def predict(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make pattern prediction using trained model
        
        Args:
            features: Input features for prediction
            
        Returns:
            Pattern prediction results
        """
        try:
            if self.model is None:
                logger.error("CNN model not trained yet")
                return None
            
            # Extract OHLCV data
            ohlcv_data = None
            for timeframe_data in features.values():
                if isinstance(timeframe_data, dict) and 'ohlcv' in timeframe_data:
                    ohlcv_data = timeframe_data['ohlcv']
                    break
            
            if ohlcv_data is None or ohlcv_data.empty:
                logger.warning("No OHLCV data available for pattern prediction")
                return None
            
            # Create chart image
            image = self._create_chart_image(ohlcv_data.tail(50))  # Use last 50 candles
            
            if image is None:
                logger.warning("Could not create chart image")
                return None
            
            # Prepare for prediction
            image_normalized = image.astype('float32') / 255.0
            image_batch = np.expand_dims(image_normalized, axis=0)
            
            # Make prediction
            prediction = self.model.predict(image_batch)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            
            # Convert back to pattern name
            predicted_pattern = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Get top 3 predictions
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            top_3_patterns = []
            
            for idx in top_3_indices:
                pattern_name = self.label_encoder.inverse_transform([idx])[0]
                pattern_confidence = float(prediction[0][idx])
                top_3_patterns.append({
                    'pattern': pattern_name,
                    'confidence': pattern_confidence
                })
            
            return {
                'predicted_pattern': predicted_pattern,
                'confidence': confidence,
                'top_3_patterns': top_3_patterns,
                'model_type': 'CNN_Pattern'
            }
            
        except Exception as e:
            logger.error(f"Error making CNN pattern prediction: {e}")
            return None
    
    def save(self, filepath: str) -> None:
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            model_path = filepath.replace('.pkl', '_model.h5')
            if self.model:
                self.model.save(model_path)
            
            # Save metadata
            model_data = {
                'label_encoder': self.label_encoder,
                'config': self.config,
                'pattern_classes': self.pattern_classes,
                'training_history': self.training_history,
                'image_size': self.image_size
            }
            
            joblib.dump(model_data, filepath)
            
            logger.info(f"CNN model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving CNN model: {e}")
    
    def load(self, filepath: str) -> None:
        """Load a trained model"""
        try:
            # Load model data
            model_data = joblib.load(filepath)
            
            self.label_encoder = model_data['label_encoder']
            self.config = model_data['config']
            self.pattern_classes = model_data['pattern_classes']
            self.training_history = model_data['training_history']
            self.image_size = model_data['image_size']
            
            # Load Keras model
            model_path = filepath.replace('.pkl', '_model.h5')
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
            
            logger.info(f"CNN model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading CNN model: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary and statistics"""
        try:
            summary = {
                'model_type': 'CNN_Pattern',
                'architecture': {
                    'image_size': self.image_size,
                    'filters': self.filters,
                    'kernel_size': self.kernel_size
                },
                'training_config': {
                    'epochs': self.epochs,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate
                },
                'pattern_classes': self.pattern_classes
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
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting CNN model summary: {e}")
            return {'error': str(e)}