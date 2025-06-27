#!/usr/bin/env python3
"""
AI Trading Signal Bot - Model Training Script
Author: baadri
Date: 2025-06-27
"""

import asyncio
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta

from data_pipeline.bybit_connector import BybitConnector
from data_pipeline.data_collector import DataCollector
from feature_engineering.technical_indicators import TechnicalIndicators
from ml_models.lstm_model import LSTMModel
from ml_models.cnn_patterns import CNNPatternModel
from ml_models.transformer_model import TransformerModel
from ml_models.xgboost_volume import XGBoostVolumeModel
from ml_models.sentiment_bert import SentimentBERTModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model training coordinator"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.bybit_connector = BybitConnector(self.config['api']['bybit'])
        self.data_collector = DataCollector(self.bybit_connector)
        self.technical_indicators = TechnicalIndicators()
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
    
    async def train_all_models(self):
        """Train all ensemble models"""
        logger.info("Starting model training process...")
        
        # 1. Collect training data
        training_data = await self._collect_training_data()
        
        # 2. Prepare features
        features = self._prepare_features(training_data)
        
        # 3. Train each model
        await self._train_lstm_model(features)
        await self._train_cnn_model(features)
        await self._train_transformer_model(features)
        await self._train_xgboost_model(features)
        await self._train_sentiment_model()
        
        logger.info("All models trained successfully!")
    
    async def _collect_training_data(self):
        """Collect historical data for training"""
        logger.info("Collecting training data...")
        
        training_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['data']['historical_days'])
        
        for pair in self.config['trading_pairs']:
            logger.info(f"Collecting data for {pair}")
            pair_data = {}
            
            for timeframe in self.config['timeframes']:
                data = await self.data_collector.get_historical_data(
                    pair, timeframe, start_date, end_date
                )
                pair_data[timeframe] = data
            
            training_data[pair] = pair_data
        
        return training_data
    
    def _prepare_features(self, training_data):
        """Prepare features for model training"""
        logger.info("Preparing features...")
        
        prepared_features = {}
        
        for pair, timeframes_data in training_data.items():
            pair_features = {}
            
            for timeframe, ohlcv_data in timeframes_data.items():
                # Calculate technical indicators
                indicators = self.technical_indicators.calculate_all(ohlcv_data)
                
                # Prepare target variable (future price movement)
                targets = self._create_targets(ohlcv_data)
                
                pair_features[timeframe] = {
                    'features': indicators,
                    'targets': targets,
                    'ohlcv': ohlcv_data
                }
            
            prepared_features[pair] = pair_features
        
        return prepared_features
    
    def _create_targets(self, ohlcv_data):
        """Create target variables for supervised learning"""
        # This would create binary targets (up/down) or regression targets
        # Implementation details would depend on the specific model requirements
        pass
    
    async def _train_lstm_model(self, features):
        """Train LSTM model"""
        logger.info("Training LSTM model...")
        
        lstm_model = LSTMModel(self.config['models']['lstm'])
        await lstm_model.train(features)
        lstm_model.save("models/lstm_model.pkl")
    
    async def _train_cnn_model(self, features):
        """Train CNN pattern recognition model"""
        logger.info("Training CNN model...")
        
        cnn_model = CNNPatternModel(self.config['models']['cnn'])
        await cnn_model.train(features)
        cnn_model.save("models/cnn_model.pkl")
    
    async def _train_transformer_model(self, features):
        """Train Transformer model"""
        logger.info("Training Transformer model...")
        
        transformer_model = TransformerModel(self.config['models']['transformer'])
        await transformer_model.train(features)
        transformer_model.save("models/transformer_model.pkl")
    
    async def _train_xgboost_model(self, features):
        """Train XGBoost volume analysis model"""
        logger.info("Training XGBoost model...")
        
        xgboost_model = XGBoostVolumeModel(self.config['models']['xgboost'])
        await xgboost_model.train(features)
        xgboost_model.save("models/xgboost_model.pkl")
    
    async def _train_sentiment_model(self):
        """Train BERT sentiment analysis model"""
        logger.info("Training BERT sentiment model...")
        
        sentiment_model = SentimentBERTModel(self.config['models']['sentiment'])
        await sentiment_model.train()
        sentiment_model.save("models/sentiment_model.pkl")

async def main():
    """Main training function"""
    trainer = ModelTrainer()
    await trainer.train_all_models()

if __name__ == "__main__":
    asyncio.run(main())
