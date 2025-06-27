"""
Real-time Data Stream
Handles real-time market data streaming
"""

import asyncio
import logging
import json
import websockets
from datetime import datetime, timezone
from typing import Dict, List, Callable, Optional, Any
import aiohttp
from dataclasses import dataclass

from .bybit_connector import BybitConnector

logger = logging.getLogger(__name__)

@dataclass
class TickerUpdate:
    """Real-time ticker update"""
    symbol: str
    price: float
    volume: float
    change: float
    change_percent: float
    timestamp: datetime

@dataclass
class TradeUpdate:
    """Real-time trade update"""
    symbol: str
    price: float
    quantity: float
    side: str  # 'buy' or 'sell'
    timestamp: datetime

class RealtimeStream:
    """Real-time market data streaming"""
    
    def __init__(self, connector: BybitConnector):
        self.connector = connector
        self.websocket = None
        self.running = False
        self.subscribers = {
            'ticker': [],
            'trade': [],
            'orderbook': [],
            'kline': []
        }
        
        # Bybit WebSocket URLs
        self.ws_urls = {
            'testnet': 'wss://stream-testnet.bybit.com/v5/public/linear',
            'mainnet': 'wss://stream.bybit.com/v5/public/linear'
        }
        
        self.subscriptions = set()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5
    
    async def start(self):
        """Start real-time data streaming"""
        logger.info("Starting real-time data stream...")
        self.running = True
        
        while self.running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                await self._connect_websocket()
                break
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                self.reconnect_attempts += 1
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    logger.info(f"Reconnecting in {self.reconnect_delay} seconds... "
                              f"(Attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    logger.error("Max reconnection attempts reached")
                    break
    
    async def stop(self):
        """Stop real-time data streaming"""
        logger.info("Stopping real-time data stream...")
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
    
    async def _connect_websocket(self):
        """Connect to Bybit WebSocket"""
        is_testnet = self.connector.config.get('testnet', True)
        ws_url = self.ws_urls['testnet' if is_testnet else 'mainnet']
        
        try:
            self.websocket = await websockets.connect(ws_url)
            logger.info(f"Connected to Bybit WebSocket: {ws_url}")
            
            # Resubscribe to all channels
            await self._resubscribe_all()
            
            # Start listening
            await self._listen_messages()
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            raise
    
    async def _listen_messages(self):
        """Listen for WebSocket messages"""
        try:
            async for message in self.websocket:
                if not self.running:
                    break
                
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing WebSocket message: {e}")
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            if self.running:
                await self._reconnect()
                
        except Exception as e:
            logger.error(f"Error in WebSocket listener: {e}")
            if self.running:
                await self._reconnect()
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        try:
            # Handle different message types
            if 'topic' in data:
                topic = data['topic']
                
                if topic.startswith('tickers'):
                    await self._handle_ticker_update(data)
                elif topic.startswith('publicTrade'):
                    await self._handle_trade_update(data)
                elif topic.startswith('orderbook'):
                    await self._handle_orderbook_update(data)
                elif topic.startswith('kline'):
                    await self._handle_kline_update(data)
            
            elif 'success' in data:
                # Subscription confirmation
                if data['success']:
                    logger.debug(f"Successfully subscribed to: {data.get('request', {}).get('args', [])}")
                else:
                    logger.warning(f"Subscription failed: {data}")
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _handle_ticker_update(self, data: Dict[str, Any]):
        """Handle ticker updates"""
        try:
            ticker_data = data.get('data', {})
            
            ticker_update = TickerUpdate(
                symbol=ticker_data.get('symbol', ''),
                price=float(ticker_data.get('lastPrice', 0)),
                volume=float(ticker_data.get('volume24h', 0)),
                change=float(ticker_data.get('price24hPcnt', 0)),
                change_percent=float(ticker_data.get('price24hPcnt', 0)) * 100,
                timestamp=datetime.now(timezone.utc)
            )
            
            # Notify subscribers
            for callback in self.subscribers['ticker']:
                try:
                    await callback(ticker_update)
                except Exception as e:
                    logger.error(f"Error in ticker callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling ticker update: {e}")
    
    async def _handle_trade_update(self, data: Dict[str, Any]):
        """Handle trade updates"""
        try:
            trade_data = data.get('data', [])
            
            for trade in trade_data:
                trade_update = TradeUpdate(
                    symbol=trade.get('s', ''),
                    price=float(trade.get('p', 0)),
                    quantity=float(trade.get('v', 0)),
                    side=trade.get('S', '').lower(),
                    timestamp=datetime.fromtimestamp(
                        int(trade.get('T', 0)) / 1000, 
                        tz=timezone.utc
                    )
                )
                
                # Notify subscribers
                for callback in self.subscribers['trade']:
                    try:
                        await callback(trade_update)
                    except Exception as e:
                        logger.error(f"Error in trade callback: {e}")
                        
        except Exception as e:
            logger.error(f"Error handling trade update: {e}")
    
    async def _handle_orderbook_update(self, data: Dict[str, Any]):
        """Handle orderbook updates"""
        try:
            # Notify subscribers with raw orderbook data
            for callback in self.subscribers['orderbook']:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Error in orderbook callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling orderbook update: {e}")
    
    async def _handle_kline_update(self, data: Dict[str, Any]):
        """Handle kline/candlestick updates"""
        try:
            # Notify subscribers with raw kline data
            for callback in self.subscribers['kline']:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Error in kline callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling kline update: {e}")
    
    async def subscribe_ticker(self, symbols: List[str], callback: Callable):
        """Subscribe to ticker updates"""
        self.subscribers['ticker'].append(callback)
        
        for symbol in symbols:
            topic = f"tickers.{symbol}"
            await self._subscribe(topic)
    
    async def subscribe_trades(self, symbols: List[str], callback: Callable):
        """Subscribe to trade updates"""
        self.subscribers['trade'].append(callback)
        
        for symbol in symbols:
            topic = f"publicTrade.{symbol}"
            await self._subscribe(topic)
    
    async def subscribe_orderbook(self, symbols: List[str], callback: Callable, depth: int = 25):
        """Subscribe to orderbook updates"""
        self.subscribers['orderbook'].append(callback)
        
        for symbol in symbols:
            topic = f"orderbook.{depth}.{symbol}"
            await self._subscribe(topic)
    
    async def subscribe_klines(self, symbols: List[str], timeframe: str, callback: Callable):
        """Subscribe to kline updates"""
        self.subscribers['kline'].append(callback)
        
        for symbol in symbols:
            topic = f"kline.{timeframe}.{symbol}"
            await self._subscribe(topic)
    
    async def _subscribe(self, topic: str):
        """Subscribe to a WebSocket topic"""
        if topic in self.subscriptions:
            return
        
        try:
            if self.websocket:
                subscription_msg = {
                    "op": "subscribe",
                    "args": [topic]
                }
                
                await self.websocket.send(json.dumps(subscription_msg))
                self.subscriptions.add(topic)
                logger.debug(f"Subscribed to: {topic}")
                
        except Exception as e:
            logger.error(f"Error subscribing to {topic}: {e}")
    
    async def _resubscribe_all(self):
        """Resubscribe to all topics after reconnection"""
        if not self.subscriptions:
            return
        
        try:
            subscription_msg = {
                "op": "subscribe",
                "args": list(self.subscriptions)
            }
            
            await self.websocket.send(json.dumps(subscription_msg))
            logger.info(f"Resubscribed to {len(self.subscriptions)} topics")
            
        except Exception as e:
            logger.error(f"Error resubscribing: {e}")
    
    async def _reconnect(self):
        """Reconnect to WebSocket"""
        logger.info("Attempting to reconnect...")
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts <= self.max_reconnect_attempts:
            await asyncio.sleep(self.reconnect_delay)
            await self._connect_websocket()
        else:
            logger.error("Max reconnection attempts reached")
            self.running = False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            'connected': self.websocket is not None and not self.websocket.closed,
            'running': self.running,
            'subscriptions': len(self.subscriptions),
            'reconnect_attempts': self.reconnect_attempts
        }