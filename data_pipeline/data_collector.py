"""
Data Collector
Collects and manages historical market data
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import sqlite3
import json
from pathlib import Path

from .bybit_connector import BybitConnector

logger = logging.getLogger(__name__)

class DataCollector:
    """Collects and manages market data"""
    
    def __init__(self, connector: BybitConnector):
        self.connector = connector
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Initialize database
        self.db_path = "data/market_data.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for data storage"""
        Path("data").mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_timestamp 
                ON ohlcv_data (symbol, timeframe, timestamp)
            """)
            
            conn.commit()
        
        logger.info("Database initialized")
    
    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        Get OHLCV data with caching
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            limit: Number of candles
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached data for {cache_key}")
            return self.cache[cache_key]['data']
        
        try:
            # Fetch from exchange
            ohlcv_data = await self.connector.fetch_ohlcv(symbol, timeframe, limit)
            
            if not ohlcv_data:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache the data
            self._cache_data(cache_key, df)
            
            # Store in database
            await self._store_ohlcv_data(symbol, timeframe, df)
            
            logger.debug(f"Collected {len(df)} candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting OHLCV data for {symbol} {timeframe}: {e}")
            
            # Try to get from database as fallback
            return await self._get_ohlcv_from_db(symbol, timeframe, limit)
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical data for a date range
        
        Args:
            symbol: Trading pair
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Calculate timeframe in milliseconds
            timeframe_ms = self._timeframe_to_ms(timeframe)
            
            # Calculate required number of candles
            duration = (end_date - start_date).total_seconds() * 1000
            required_candles = int(duration / timeframe_ms)
            
            all_data = []
            current_start = start_date
            
            while current_start < end_date:
                # Fetch batch of data
                batch_limit = min(1000, required_candles)  # Max 1000 per request
                
                # Convert to timestamp
                since = int(current_start.timestamp() * 1000)
                
                ohlcv_batch = await self.connector.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=batch_limit
                )
                
                if not ohlcv_batch:
                    break
                
                all_data.extend(ohlcv_batch)
                
                # Update start time for next batch
                last_timestamp = ohlcv_batch[-1][0]
                current_start = datetime.fromtimestamp(last_timestamp / 1000, tz=timezone.utc)
                current_start += timedelta(milliseconds=timeframe_ms)
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
                required_candles -= len(ohlcv_batch)
                if required_candles <= 0:
                    break
            
            if not all_data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(
                all_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            logger.info(f"Collected {len(df)} historical candles for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting historical data: {e}")
            return pd.DataFrame()
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds"""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }
        return timeframe_map.get(timeframe, 60 * 1000)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_entry = self.cache[cache_key]
        cache_age = (datetime.now(timezone.utc) - cache_entry['timestamp']).total_seconds()
        
        return cache_age < self.cache_duration
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame):
        """Cache data with timestamp"""
        self.cache[cache_key] = {
            'data': data.copy(),
            'timestamp': datetime.now(timezone.utc)
        }
    
    async def _store_ohlcv_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Store OHLCV data in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for timestamp, row in df.iterrows():
                    conn.execute("""
                        INSERT OR REPLACE INTO ohlcv_data 
                        (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        timeframe,
                        int(timestamp.timestamp() * 1000),
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume'])
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing OHLCV data: {e}")
    
    async def _get_ohlcv_from_db(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Get OHLCV data from database as fallback"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv_data
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=(symbol, timeframe, limit),
                    index_col='timestamp'
                )
                
                if not df.empty:
                    df.index = pd.to_datetime(df.index, unit='ms')
                    df = df.sort_index()
                
                logger.info(f"Retrieved {len(df)} candles from database for {symbol} {timeframe}")
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving data from database: {e}")
            return pd.DataFrame()
    
    async def get_multiple_timeframes(
        self, 
        symbol: str, 
        timeframes: List[str], 
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes simultaneously"""
        tasks = [
            self.get_ohlcv(symbol, tf, limit) 
            for tf in timeframes
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for timeframe, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.error(f"Error getting data for {symbol} {timeframe}: {result}")
                data[timeframe] = pd.DataFrame()
            else:
                data[timeframe] = result
        
        return data