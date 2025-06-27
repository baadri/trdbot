"""
Data Validator
Validates and cleans market data
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates and cleans market data"""
    
    def __init__(self):
        self.validation_rules = {
            'ohlcv': {
                'required_columns': ['open', 'high', 'low', 'close', 'volume'],
                'positive_columns': ['open', 'high', 'low', 'close', 'volume'],
                'ohlc_rules': True,  # high >= open,close,low; low <= open,close,high
                'max_price_change': 0.5,  # 50% max change between candles
                'min_volume': 0,
                'max_gap_minutes': 60  # Max gap between candles in minutes
            },
            'ticker': {
                'required_fields': ['symbol', 'price', 'volume'],
                'positive_fields': ['price', 'volume'],
                'max_age_seconds': 300  # 5 minutes max age
            },
            'orderbook': {
                'required_fields': ['bids', 'asks'],
                'min_levels': 10,
                'max_spread_percent': 0.1  # 10% max spread
            }
        }
    
    def validate_ohlcv(self, df: pd.DataFrame, symbol: str = "") -> bool:
        """
        Validate OHLCV DataFrame
        
        Args:
            df: OHLCV DataFrame
            symbol: Symbol name for logging
            
        Returns:
            True if data is valid, False otherwise
        """
        if df is None or df.empty:
            logger.warning(f"Empty OHLCV data for {symbol}")
            return False
        
        try:
            rules = self.validation_rules['ohlcv']
            
            # Check required columns
            missing_cols = set(rules['required_columns']) - set(df.columns)
            if missing_cols:
                logger.error(f"Missing columns in OHLCV data for {symbol}: {missing_cols}")
                return False
            
            # Check for NaN values
            if df[rules['required_columns']].isnull().any().any():
                logger.warning(f"NaN values found in OHLCV data for {symbol}")
                return False
            
            # Check positive values
            for col in rules['positive_columns']:
                if (df[col] <= 0).any():
                    logger.warning(f"Non-positive values in column {col} for {symbol}")
                    return False
            
            # Check OHLC rules
            if rules['ohlc_rules']:
                if not self._validate_ohlc_rules(df, symbol):
                    return False
            
            # Check for extreme price changes
            if not self._validate_price_changes(df, symbol, rules['max_price_change']):
                return False
            
            # Check timestamp gaps
            if not self._validate_timestamp_gaps(df, symbol, rules['max_gap_minutes']):
                return False
            
            logger.debug(f"OHLCV data validation passed for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating OHLCV data for {symbol}: {e}")
            return False
    
    def _validate_ohlc_rules(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate OHLC relationship rules"""
        try:
            # High should be >= open, close, low
            invalid_high = (
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['high'] < df['low'])
            ).any()
            
            # Low should be <= open, close, high
            invalid_low = (
                (df['low'] > df['open']) |
                (df['low'] > df['close']) |
                (df['low'] > df['high'])
            ).any()
            
            if invalid_high or invalid_low:
                logger.warning(f"Invalid OHLC relationships found for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating OHLC rules for {symbol}: {e}")
            return False
    
    def _validate_price_changes(self, df: pd.DataFrame, symbol: str, max_change: float) -> bool:
        """Validate price changes between candles"""
        try:
            if len(df) < 2:
                return True
            
            # Calculate price changes
            close_changes = df['close'].pct_change().abs()
            
            # Check for extreme changes
            extreme_changes = close_changes > max_change
            
            if extreme_changes.any():
                num_extreme = extreme_changes.sum()
                logger.warning(f"Found {num_extreme} extreme price changes for {symbol}")
                
                # Allow some extreme changes (market events), but not too many
                if num_extreme > len(df) * 0.05:  # More than 5% of candles
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating price changes for {symbol}: {e}")
            return False
    
    def _validate_timestamp_gaps(self, df: pd.DataFrame, symbol: str, max_gap_minutes: int) -> bool:
        """Validate timestamp gaps"""
        try:
            if len(df) < 2:
                return True
            
            # Calculate time differences
            time_diffs = df.index.to_series().diff()
            
            # Find large gaps
            max_gap = timedelta(minutes=max_gap_minutes)
            large_gaps = time_diffs > max_gap
            
            if large_gaps.any():
                num_gaps = large_gaps.sum()
                logger.warning(f"Found {num_gaps} large timestamp gaps for {symbol}")
                
                # Allow some gaps, but not too many
                if num_gaps > len(df) * 0.1:  # More than 10% of candles
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating timestamp gaps for {symbol}: {e}")
            return False
    
    def validate_ticker(self, ticker_data: Dict[str, Any], symbol: str = "") -> bool:
        """Validate ticker data"""
        if not ticker_data:
            logger.warning(f"Empty ticker data for {symbol}")
            return False
        
        try:
            rules = self.validation_rules['ticker']
            
            # Check required fields
            missing_fields = set(rules['required_fields']) - set(ticker_data.keys())
            if missing_fields:
                logger.error(f"Missing fields in ticker data for {symbol}: {missing_fields}")
                return False
            
            # Check positive values
            for field in rules['positive_fields']:
                if ticker_data.get(field, 0) <= 0:
                    logger.warning(f"Non-positive value in field {field} for {symbol}")
                    return False
            
            # Check data age
            if 'timestamp' in ticker_data:
                age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                    ticker_data['timestamp'] / 1000, tz=timezone.utc
                )
                
                if age.total_seconds() > rules['max_age_seconds']:
                    logger.warning(f"Ticker data too old for {symbol}: {age.total_seconds()}s")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating ticker data for {symbol}: {e}")
            return False
    
    def validate_orderbook(self, orderbook_data: Dict[str, Any], symbol: str = "") -> bool:
        """Validate orderbook data"""
        if not orderbook_data:
            logger.warning(f"Empty orderbook data for {symbol}")
            return False
        
        try:
            rules = self.validation_rules['orderbook']
            
            # Check required fields
            if 'bids' not in orderbook_data or 'asks' not in orderbook_data:
                logger.error(f"Missing bids/asks in orderbook data for {symbol}")
                return False
            
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']
            
            # Check minimum levels
            if len(bids) < rules['min_levels'] or len(asks) < rules['min_levels']:
                logger.warning(f"Insufficient orderbook levels for {symbol}")
                return False
            
            # Check bid/ask order
            if not self._validate_orderbook_order(bids, asks):
                logger.warning(f"Invalid orderbook order for {symbol}")
                return False
            
            # Check spread
            if not self._validate_spread(bids, asks, symbol, rules['max_spread_percent']):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating orderbook data for {symbol}: {e}")
            return False
    
    def _validate_orderbook_order(self, bids: List[List], asks: List[List]) -> bool:
        """Validate orderbook price ordering"""
        try:
            # Bids should be in descending order
            bid_prices = [float(bid[0]) for bid in bids]
            if bid_prices != sorted(bid_prices, reverse=True):
                return False
            
            # Asks should be in ascending order
            ask_prices = [float(ask[0]) for ask in asks]
            if ask_prices != sorted(ask_prices):
                return False
            
            # Best bid should be less than best ask
            if bid_prices[0] >= ask_prices[0]:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating orderbook order: {e}")
            return False
    
    def _validate_spread(self, bids: List[List], asks: List[List], symbol: str, max_spread_percent: float) -> bool:
        """Validate bid-ask spread"""
        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            spread_percent = (best_ask - best_bid) / best_bid
            
            if spread_percent > max_spread_percent:
                logger.warning(f"Large spread for {symbol}: {spread_percent:.4f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating spread for {symbol}: {e}")
            return False
    
    def clean_ohlcv(self, df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        """Clean and fix OHLCV data"""
        if df is None or df.empty:
            return df
        
        try:
            cleaned_df = df.copy()
            
            # Remove rows with NaN values
            cleaned_df = cleaned_df.dropna()
            
            # Remove duplicate timestamps
            cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep='first')]
            
            # Fix OHLC relationships
            cleaned_df = self._fix_ohlc_relationships(cleaned_df)
            
            # Remove extreme outliers
            cleaned_df = self._remove_price_outliers(cleaned_df)
            
            # Forward fill small gaps
            cleaned_df = self._fill_small_gaps(cleaned_df)
            
            logger.debug(f"Cleaned OHLCV data for {symbol}: {len(df)} -> {len(cleaned_df)} rows")
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning OHLCV data for {symbol}: {e}")
            return df
    
    def _fix_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix OHLC relationship violations"""
        try:
            fixed_df = df.copy()
            
            # Ensure high is the maximum of open, high, low, close
            fixed_df['high'] = fixed_df[['open', 'high', 'low', 'close']].max(axis=1)
            
            # Ensure low is the minimum of open, high, low, close
            fixed_df['low'] = fixed_df[['open', 'high', 'low', 'close']].min(axis=1)
            
            return fixed_df
            
        except Exception as e:
            logger.error(f"Error fixing OHLC relationships: {e}")
            return df
    
    def _remove_price_outliers(self, df: pd.DataFrame, z_threshold: float = 5.0) -> pd.DataFrame:
        """Remove extreme price outliers using z-score"""
        try:
            if len(df) < 10:
                return df
            
            cleaned_df = df.copy()
            
            # Calculate z-scores for close prices
            close_returns = cleaned_df['close'].pct_change()
            z_scores = np.abs((close_returns - close_returns.mean()) / close_returns.std())
            
            # Remove outliers
            outlier_mask = z_scores > z_threshold
            cleaned_df = cleaned_df[~outlier_mask]
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            return df
    
    def _fill_small_gaps(self, df: pd.DataFrame, max_gap_minutes: int = 15) -> pd.DataFrame:
        """Forward fill small timestamp gaps"""
        try:
            # This would implement gap filling logic
            # For now, just return the DataFrame as-is
            return df
            
        except Exception as e:
            logger.error(f"Error filling gaps: {e}")
            return df
    
    def get_data_quality_score(self, df: pd.DataFrame, symbol: str = "") -> float:
        """Calculate data quality score (0-1)"""
        try:
            if df is None or df.empty:
                return 0.0
            
            score = 1.0
            
            # Penalize for missing data
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            score -= missing_ratio * 0.3
            
            # Penalize for OHLC violations
            violations = self._count_ohlc_violations(df)
            violation_ratio = violations / len(df)
            score -= violation_ratio * 0.2
            
            # Penalize for extreme changes
            extreme_changes = self._count_extreme_changes(df)
            extreme_ratio = extreme_changes / len(df)
            score -= extreme_ratio * 0.2
            
            # Penalize for large gaps
            large_gaps = self._count_large_gaps(df)
            gap_ratio = large_gaps / len(df)
            score -= gap_ratio * 0.3
            
            return max(0.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating data quality score for {symbol}: {e}")
            return 0.0
    
    def _count_ohlc_violations(self, df: pd.DataFrame) -> int:
        """Count OHLC relationship violations"""
        try:
            violations = (
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['high'] < df['low']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close']) |
                (df['low'] > df['high'])
            ).sum()
            
            return violations
            
        except Exception:
            return 0
    
    def _count_extreme_changes(self, df: pd.DataFrame, threshold: float = 0.2) -> int:
        """Count extreme price changes"""
        try:
            if len(df) < 2:
                return 0
            
            changes = df['close'].pct_change().abs()
            extreme_changes = (changes > threshold).sum()
            
            return extreme_changes
            
        except Exception:
            return 0
    
    def _count_large_gaps(self, df: pd.DataFrame, max_minutes: int = 60) -> int:
        """Count large timestamp gaps"""
        try:
            if len(df) < 2:
                return 0
            
            time_diffs = df.index.to_series().diff()
            max_gap = timedelta(minutes=max_minutes)
            large_gaps = (time_diffs > max_gap).sum()
            
            return large_gaps
            
        except Exception:
            return 0