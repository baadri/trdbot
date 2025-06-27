"""
Bybit Exchange API Connector
Handles connection and API calls to Bybit exchange
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
import ccxt.async_support as ccxt
from datetime import datetime, timezone
import aiohttp

logger = logging.getLogger(__name__)

class BybitConnector:
    """Bybit exchange connector with async support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchange = None
        self.session = None
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize CCXT exchange instance"""
        try:
            self.exchange = ccxt.bybit({
                'apiKey': self.config['api_key'],
                'secret': self.config['api_secret'],
                'sandbox': self.config.get('testnet', True),
                'enableRateLimit': True,
                'rateLimit': 100,  # 100ms
                'options': {
                    'recvWindow': 10000,
                    'timeDifference': 0,
                }
            })
            
            logger.info("Bybit exchange connector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Bybit connector: {e}")
            raise
    
    async def connect(self):
        """Establish connection to exchange"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Test connection
            await self.exchange.load_markets()
            
            # Test API credentials
            balance = await self.exchange.fetch_balance()
            
            logger.info("Successfully connected to Bybit")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Bybit: {e}")
            return False
    
    async def disconnect(self):
        """Close connection"""
        try:
            if self.exchange:
                await self.exchange.close()
            
            if self.session:
                await self.session.close()
            
            logger.info("Disconnected from Bybit")
            
        except Exception as e:
            logger.error(f"Error disconnecting from Bybit: {e}")
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> List[List]:
        """
        Fetch OHLCV data for a symbol
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            List of OHLCV data [[timestamp, open, high, low, close, volume], ...]
        """
        await self._rate_limit()
        
        try:
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            logger.debug(f"Fetched {len(ohlcv)} candles for {symbol} {timeframe}")
            return ohlcv
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} {timeframe}: {e}")
            return []
    
    async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch current ticker data"""
        await self._rate_limit()
        
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    async def fetch_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """Fetch order book data"""
        await self._rate_limit()
        
        try:
            order_book = await self.exchange.fetch_order_book(symbol, limit)
            return order_book
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None
    
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Fetch recent trades"""
        await self._rate_limit()
        
        try:
            trades = await self.exchange.fetch_trades(symbol, limit=limit)
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            return []
    
    async def fetch_balance(self) -> Optional[Dict]:
        """Fetch account balance"""
        await self._rate_limit()
        
        try:
            balance = await self.exchange.fetch_balance()
            return balance
            
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return None
    
    async def get_server_time(self) -> Optional[int]:
        """Get server timestamp"""
        try:
            # Use exchange's built-in method if available
            if hasattr(self.exchange, 'fetch_time'):
                server_time = await self.exchange.fetch_time()
                return server_time
            else:
                # Fallback to current time
                return int(datetime.now(timezone.utc).timestamp() * 1000)
                
        except Exception as e:
            logger.error(f"Error getting server time: {e}")
            return None
    
    async def check_connection(self) -> bool:
        """Check if connection is alive"""
        try:
            server_time = await self.get_server_time()
            return server_time is not None
            
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False