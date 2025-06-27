#!/usr/bin/env python3
"""
AI Trading Signal Bot - Main Application
Author: baadri
Date: 2025-06-27
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dataclasses import dataclass

# Import custom modules
from data_pipeline.bybit_connector import BybitConnector
from data_pipeline.data_collector import DataCollector
from data_pipeline.realtime_stream import RealtimeStream
from data_pipeline.data_validator import DataValidator

from feature_engineering.technical_indicators import TechnicalIndicators
from feature_engineering.market_microstructure import MarketMicrostructure
from feature_engineering.sentiment_scraper import SentimentScraper
from feature_engineering.feature_selector import FeatureSelector

from ml_models.ensemble_voting import EnsembleModel
from trading_logic.signal_generator import SignalGenerator
from trading_logic.risk_manager import RiskManager
from trading_logic.performance_tracker import PerformanceTracker

from notifications.telegram_bot import TelegramBot
from notifications.alert_manager import AlertManager
from visualization.dashboard import Dashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    action: str  # BUY/SELL
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    confidence: float
    timeframe: str
    reasons: List[str]
    risk_amount: float
    position_size: float
    chart_url: Optional[str] = None
    timestamp: Optional[datetime] = None

class TradingBot:
    """Main Trading Bot Class"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.trading_params = self._load_config("config/trading_params.yaml")
        self.running = False
        
        # Initialize components
        self._initialize_components()
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, path: str) -> Dict:
        """Load YAML configuration file"""
        try:
            with open(path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {path}: {e}")
            sys.exit(1)
    
    def _initialize_components(self):
        """Initialize all bot components"""
        logger.info("Initializing Trading Bot components...")
        
        # Data Pipeline
        self.bybit_connector = BybitConnector(self.config['api']['bybit'])
        self.data_collector = DataCollector(self.bybit_connector)
        self.realtime_stream = RealtimeStream(self.bybit_connector)
        self.data_validator = DataValidator()
        
        # Feature Engineering
        self.technical_indicators = TechnicalIndicators(
            self.trading_params['technical_indicators']
        )
        self.market_microstructure = MarketMicrostructure()
        self.sentiment_scraper = SentimentScraper()
        self.feature_selector = FeatureSelector()
        
        # ML Models
        self.ensemble_model = EnsembleModel(self.config['models'])
        
        # Trading Logic
        self.signal_generator = SignalGenerator(
            self.ensemble_model,
            self.trading_params['signal_generation']
        )
        self.risk_manager = RiskManager(self.trading_params['risk_management'])
        self.performance_tracker = PerformanceTracker()
        
        # Notifications
        self.telegram_bot = TelegramBot(self.config['api']['telegram'])
        self.alert_manager = AlertManager()
        
        # Dashboard
        self.dashboard = Dashboard(
            self.config['notifications']['dashboard']
        )
        
        logger.info("All components initialized successfully")
    
    async def start(self):
        """Start the trading bot"""
        logger.info("Starting AI Trading Signal Bot...")
        self.running = True
        
        try:
            # Start dashboard in background
            await self.dashboard.start()
            
            # Start main trading loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop the trading bot gracefully"""
        logger.info("Stopping Trading Bot...")
        self.running = False
        
        # Stop all components
        await self.dashboard.stop()
        await self.telegram_bot.stop()
        
        logger.info("Trading Bot stopped successfully")
    
    async def _main_loop(self):
        """Main trading loop"""
        logger.info("Starting main trading loop...")
        
        while self.running:
            try:
                # 1. Collect market data for all trading pairs
                market_data = await self._collect_market_data()
                
                # 2. Calculate technical indicators and features
                features = await self._calculate_features(market_data)
                
                # 3. Analyze market sentiment
                sentiment_data = await self._analyze_sentiment()
                
                # 4. Generate trading signals
                signals = await self._generate_signals(features, sentiment_data)
                
                # 5. Filter and validate signals
                validated_signals = await self._validate_signals(signals)
                
                # 6. Process and send signals
                await self._process_signals(validated_signals)
                
                # 7. Update performance metrics
                await self._update_performance()
                
                # 8. Send daily reports if needed
                await self._send_daily_reports()
                
                # Wait before next iteration
                await asyncio.sleep(self.config['data']['update_interval'])
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _collect_market_data(self) -> Dict:
        """Collect market data for all trading pairs"""
        logger.debug("Collecting market data...")
        
        market_data = {}
        
        for pair in self.config['trading_pairs']:
            try:
                # Collect data for all timeframes
                pair_data = {}
                for timeframe in self.config['timeframes']:
                    data = await self.data_collector.get_ohlcv(
                        pair, timeframe, limit=500
                    )
                    
                    # Validate data
                    if self.data_validator.validate_ohlcv(data):
                        pair_data[timeframe] = data
                    else:
                        logger.warning(f"Invalid data for {pair} {timeframe}")
                
                market_data[pair] = pair_data
                
            except Exception as e:
                logger.error(f"Error collecting data for {pair}: {e}")
        
        return market_data
    
    async def _calculate_features(self, market_data: Dict) -> Dict:
        """Calculate technical indicators and features"""
        logger.debug("Calculating technical features...")
        
        features = {}
        
        for pair, timeframes_data in market_data.items():
            try:
                pair_features = {}
                
                for timeframe, ohlcv_data in timeframes_data.items():
                    # Calculate technical indicators
                    indicators = self.technical_indicators.calculate_all(
                        ohlcv_data
                    )
                    
                    # Calculate market microstructure features
                    microstructure = await self.market_microstructure.analyze(
                        pair, timeframe
                    )
                    
                    # Combine features
                    pair_features[timeframe] = {
                        'indicators': indicators,
                        'microstructure': microstructure,
                        'ohlcv': ohlcv_data
                    }
                
                # Select best features
                selected_features = self.feature_selector.select_features(
                    pair_features
                )
                
                features[pair] = selected_features
                
            except Exception as e:
                logger.error(f"Error calculating features for {pair}: {e}")
        
        return features
    
    async def _analyze_sentiment(self) -> Dict:
        """Analyze market sentiment from various sources"""
        logger.debug("Analyzing market sentiment...")
        
        try:
            sentiment_data = await self.sentiment_scraper.get_market_sentiment()
            return sentiment_data
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {}
    
    async def _generate_signals(self, features: Dict, sentiment_data: Dict) -> List[TradingSignal]:
        """Generate trading signals using ensemble model"""
        logger.debug("Generating trading signals...")
        
        signals = []
        
        try:
            for pair, pair_features in features.items():
                # Generate signal for this pair
                signal = await self.signal_generator.generate_signal(
                    pair, pair_features, sentiment_data
                )
                
                if signal:
                    signals.append(signal)
            
            logger.info(f"Generated {len(signals)} potential signals")
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    async def _validate_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Validate and filter signals based on risk management"""
        logger.debug("Validating trading signals...")
        
        validated_signals = []
        
        for signal in signals:
            try:
                # Risk management validation
                if self.risk_manager.validate_signal(signal):
                    # Calculate position size
                    signal.position_size = self.risk_manager.calculate_position_size(
                        signal
                    )
                    
                    validated_signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error validating signal for {signal.symbol}: {e}")
        
        logger.info(f"Validated {len(validated_signals)} signals")
        return validated_signals
    
    async def _process_signals(self, signals: List[TradingSignal]):
        """Process and send validated signals"""
        for signal in signals:
            try:
                # Add timestamp
                signal.timestamp = datetime.now(timezone.utc)
                
                # Generate chart
                signal.chart_url = await self._generate_chart(signal)
                
                # Send to Telegram
                await self.telegram_bot.send_signal(signal)
                
                # Update dashboard
                await self.dashboard.add_signal(signal)
                
                # Track performance
                self.performance_tracker.add_signal(signal)
                
                logger.info(f"Processed signal: {signal.symbol} {signal.action}")
                
            except Exception as e:
                logger.error(f"Error processing signal {signal.symbol}: {e}")
    
    async def _generate_chart(self, signal: TradingSignal) -> Optional[str]:
        """Generate chart for the signal"""
        # This would be implemented in visualization module
        # For now, return None
        return None
    
    async def _update_performance(self):
        """Update performance metrics"""
        try:
            await self.performance_tracker.update_metrics()
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    async def _send_daily_reports(self):
        """Send daily performance reports"""
        try:
            current_hour = datetime.now(timezone.utc).hour
            report_hour = int(self.config['notifications']['telegram']['summary_time'].split(':')[0])
            
            if current_hour == report_hour:
                report = await self.performance_tracker.generate_daily_report()
                await self.telegram_bot.send_daily_report(report)
                
        except Exception as e:
            logger.error(f"Error sending daily report: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

async def main():
    """Main entry point"""
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    # Initialize and start bot
    bot = TradingBot()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
