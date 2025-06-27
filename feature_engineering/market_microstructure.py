"""
Market Microstructure Analysis
Analyzes order book, trade flow, and market depth
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import asyncio

logger = logging.getLogger(__name__)

class MarketMicrostructure:
    """Market microstructure analyzer"""
    
    def __init__(self):
        self.order_flow_cache = {}
        self.imbalance_cache = {}
        self.cache_duration = 60  # seconds
    
    async def analyze(self, symbol: str, timeframe: str, connector=None) -> Dict[str, Any]:
        """
        Analyze market microstructure for a symbol
        
        Args:
            symbol: Trading pair
            timeframe: Analysis timeframe
            connector: Exchange connector for real-time data
            
        Returns:
            Dictionary with microstructure analysis
        """
        analysis = {}
        
        try:
            if connector:
                # Get real-time order book
                orderbook = await connector.fetch_order_book(symbol, limit=100)
                if orderbook:
                    analysis.update(self.analyze_order_book(orderbook))
                
                # Get recent trades
                trades = await connector.fetch_trades(symbol, limit=100)
                if trades:
                    analysis.update(self.analyze_trade_flow(trades))
                
                # Calculate order flow imbalance
                imbalance = await self.calculate_order_flow_imbalance(symbol, connector)
                if imbalance:
                    analysis.update(imbalance)
            
            logger.debug(f"Market microstructure analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in market microstructure analysis for {symbol}: {e}")
            return {}
    
    def analyze_order_book(self, orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze order book structure
        
        Args:
            orderbook: Order book data with bids and asks
            
        Returns:
            Dictionary with order book analysis
        """
        analysis = {}
        
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return analysis
            
            # Basic spread analysis
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            analysis['bid_price'] = best_bid
            analysis['ask_price'] = best_ask
            analysis['spread'] = best_ask - best_bid
            analysis['spread_percent'] = (analysis['spread'] / best_bid) * 100
            
            # Order book depth analysis
            analysis.update(self._analyze_depth(bids, asks))
            
            # Order book imbalance
            analysis.update(self._calculate_book_imbalance(bids, asks))
            
            # Resistance and support levels from order book
            analysis.update(self._identify_book_levels(bids, asks))
            
            # Market impact estimation
            analysis.update(self._estimate_market_impact(bids, asks))
            
        except Exception as e:
            logger.error(f"Error analyzing order book: {e}")
        
        return analysis
    
    def _analyze_depth(self, bids: List[List], asks: List[List], levels: int = 10) -> Dict[str, Any]:
        """Analyze order book depth"""
        depth_analysis = {}
        
        try:
            # Calculate cumulative volumes
            bid_volumes = [float(bid[1]) for bid in bids[:levels]]
            ask_volumes = [float(ask[1]) for ask in asks[:levels]]
            
            depth_analysis['bid_volume_total'] = sum(bid_volumes)
            depth_analysis['ask_volume_total'] = sum(ask_volumes)
            depth_analysis['total_depth'] = depth_analysis['bid_volume_total'] + depth_analysis['ask_volume_total']
            
            # Depth ratio
            if depth_analysis['ask_volume_total'] > 0:
                depth_analysis['bid_ask_volume_ratio'] = (
                    depth_analysis['bid_volume_total'] / depth_analysis['ask_volume_total']
                )
            
            # Average order size
            depth_analysis['avg_bid_size'] = np.mean(bid_volumes) if bid_volumes else 0
            depth_analysis['avg_ask_size'] = np.mean(ask_volumes) if ask_volumes else 0
            
            # Order size distribution
            depth_analysis['bid_size_std'] = np.std(bid_volumes) if len(bid_volumes) > 1 else 0
            depth_analysis['ask_size_std'] = np.std(ask_volumes) if len(ask_volumes) > 1 else 0
            
            # Large order identification
            if bid_volumes:
                bid_threshold = np.mean(bid_volumes) + 2 * np.std(bid_volumes)
                depth_analysis['large_bid_orders'] = len([v for v in bid_volumes if v > bid_threshold])
            
            if ask_volumes:
                ask_threshold = np.mean(ask_volumes) + 2 * np.std(ask_volumes)
                depth_analysis['large_ask_orders'] = len([v for v in ask_volumes if v > ask_threshold])
        
        except Exception as e:
            logger.error(f"Error analyzing depth: {e}")
        
        return depth_analysis
    
    def _calculate_book_imbalance(self, bids: List[List], asks: List[List], levels: int = 5) -> Dict[str, Any]:
        """Calculate order book imbalance"""
        imbalance = {}
        
        try:
            # Calculate imbalance for top levels
            bid_volume = sum(float(bid[1]) for bid in bids[:levels])
            ask_volume = sum(float(ask[1]) for ask in asks[:levels])
            
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                imbalance['order_book_imbalance'] = (bid_volume - ask_volume) / total_volume
                imbalance['bid_dominance'] = bid_volume / total_volume
                imbalance['ask_dominance'] = ask_volume / total_volume
            
            # Weighted imbalance (price-weighted)
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
            
            weighted_bid_volume = 0
            weighted_ask_volume = 0
            
            for bid in bids[:levels]:
                price, volume = float(bid[0]), float(bid[1])
                weight = 1 - abs(price - mid_price) / mid_price
                weighted_bid_volume += volume * weight
            
            for ask in asks[:levels]:
                price, volume = float(ask[0]), float(ask[1])
                weight = 1 - abs(price - mid_price) / mid_price
                weighted_ask_volume += volume * weight
            
            total_weighted = weighted_bid_volume + weighted_ask_volume
            if total_weighted > 0:
                imbalance['weighted_imbalance'] = (weighted_bid_volume - weighted_ask_volume) / total_weighted
        
        except Exception as e:
            logger.error(f"Error calculating book imbalance: {e}")
        
        return imbalance
    
    def _identify_book_levels(self, bids: List[List], asks: List[List]) -> Dict[str, Any]:
        """Identify significant support/resistance levels from order book"""
        levels = {}
        
        try:
            # Find large orders that could act as support/resistance
            bid_volumes = [(float(bid[0]), float(bid[1])) for bid in bids]
            ask_volumes = [(float(ask[0]), float(ask[1])) for ask in asks]
            
            # Sort by volume to find largest orders
            bid_volumes.sort(key=lambda x: x[1], reverse=True)
            ask_volumes.sort(key=lambda x: x[1], reverse=True)
            
            # Top support levels (large bids)
            if bid_volumes:
                levels['strong_support'] = bid_volumes[0][0]  # Largest bid
                levels['support_volume'] = bid_volumes[0][1]
            
            # Top resistance levels (large asks)
            if ask_volumes:
                levels['strong_resistance'] = ask_volumes[0][0]  # Largest ask
                levels['resistance_volume'] = ask_volumes[0][1]
            
            # Cluster analysis for price levels
            levels.update(self._find_price_clusters(bids, asks))
        
        except Exception as e:
            logger.error(f"Error identifying book levels: {e}")
        
        return levels
    
    def _find_price_clusters(self, bids: List[List], asks: List[List], threshold: float = 0.001) -> Dict[str, Any]:
        """Find price levels with clustered orders"""
        clusters = {}
        
        try:
            # Combine all prices
            all_prices = []
            all_prices.extend([float(bid[0]) for bid in bids])
            all_prices.extend([float(ask[0]) for ask in asks])
            
            if not all_prices:
                return clusters
            
            # Simple clustering by proximity
            all_prices.sort()
            price_clusters = []
            current_cluster = [all_prices[0]]
            
            for i in range(1, len(all_prices)):
                if abs(all_prices[i] - all_prices[i-1]) / all_prices[i-1] <= threshold:
                    current_cluster.append(all_prices[i])
                else:
                    if len(current_cluster) >= 3:  # Significant cluster
                        price_clusters.append(current_cluster)
                    current_cluster = [all_prices[i]]
            
            # Add final cluster
            if len(current_cluster) >= 3:
                price_clusters.append(current_cluster)
            
            # Identify most significant clusters
            if price_clusters:
                largest_cluster = max(price_clusters, key=len)
                clusters['price_cluster_center'] = np.mean(largest_cluster)
                clusters['price_cluster_size'] = len(largest_cluster)
        
        except Exception as e:
            logger.error(f"Error finding price clusters: {e}")
        
        return clusters
    
    def _estimate_market_impact(self, bids: List[List], asks: List[List]) -> Dict[str, Any]:
        """Estimate market impact for different order sizes"""
        impact = {}
        
        try:
            # Calculate cumulative volumes and prices
            bid_cumulative = []
            ask_cumulative = []
            
            bid_vol_sum = 0
            ask_vol_sum = 0
            
            for bid in bids:
                price, volume = float(bid[0]), float(bid[1])
                bid_vol_sum += volume
                bid_cumulative.append((price, bid_vol_sum))
            
            for ask in asks:
                price, volume = float(ask[0]), float(ask[1])
                ask_vol_sum += volume
                ask_cumulative.append((price, ask_vol_sum))
            
            # Estimate impact for different order sizes
            order_sizes = [1000, 5000, 10000, 50000]  # USDT values
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            
            for size in order_sizes:
                # Buy order impact
                buy_impact = self._calculate_order_impact(ask_cumulative, size / best_ask, 'buy')
                if buy_impact:
                    impact[f'buy_impact_{size}'] = buy_impact
                
                # Sell order impact
                sell_impact = self._calculate_order_impact(bid_cumulative, size / best_bid, 'sell')
                if sell_impact:
                    impact[f'sell_impact_{size}'] = sell_impact
        
        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
        
        return impact
    
    def _calculate_order_impact(self, cumulative_data: List[Tuple[float, float]], order_volume: float, side: str) -> Optional[float]:
        """Calculate price impact for a given order volume"""
        try:
            if not cumulative_data or order_volume <= 0:
                return None
            
            # Find where the order would be filled
            for i, (price, cum_volume) in enumerate(cumulative_data):
                if cum_volume >= order_volume:
                    # Calculate average fill price
                    if i == 0:
                        avg_fill_price = price
                    else:
                        # Weighted average of prices up to this point
                        total_value = 0
                        total_volume = 0
                        remaining_volume = order_volume
                        
                        for j in range(i + 1):
                            level_price = cumulative_data[j][0]
                            level_volume = cumulative_data[j][1] - (cumulative_data[j-1][1] if j > 0 else 0)
                            
                            volume_to_use = min(remaining_volume, level_volume)
                            total_value += level_price * volume_to_use
                            total_volume += volume_to_use
                            remaining_volume -= volume_to_use
                            
                            if remaining_volume <= 0:
                                break
                        
                        avg_fill_price = total_value / total_volume if total_volume > 0 else price
                    
                    # Calculate impact as percentage
                    reference_price = cumulative_data[0][0]
                    impact = abs(avg_fill_price - reference_price) / reference_price
                    
                    return impact
            
            return None  # Order too large for available liquidity
            
        except Exception as e:
            logger.error(f"Error calculating order impact: {e}")
            return None
    
    def analyze_trade_flow(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze recent trade flow"""
        analysis = {}
        
        try:
            if not trades:
                return analysis
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(trades)
            
            # Basic trade statistics
            analysis['total_trades'] = len(trades)
            analysis['total_volume'] = df['amount'].sum() if 'amount' in df.columns else 0
            analysis['avg_trade_size'] = df['amount'].mean() if 'amount' in df.columns else 0
            
            # Buy vs Sell analysis
            if 'side' in df.columns:
                buy_trades = df[df['side'] == 'buy']
                sell_trades = df[df['side'] == 'sell']
                
                analysis['buy_trade_count'] = len(buy_trades)
                analysis['sell_trade_count'] = len(sell_trades)
                analysis['buy_volume'] = buy_trades['amount'].sum() if len(buy_trades) > 0 else 0
                analysis['sell_volume'] = sell_trades['amount'].sum() if len(sell_trades) > 0 else 0
                
                # Trade flow bias
                total_volume = analysis['buy_volume'] + analysis['sell_volume']
                if total_volume > 0:
                    analysis['buy_pressure'] = analysis['buy_volume'] / total_volume
                    analysis['sell_pressure'] = analysis['sell_volume'] / total_volume
                    analysis['flow_bias'] = analysis['buy_pressure'] - analysis['sell_pressure']
            
            # Trade size distribution
            if 'amount' in df.columns:
                analysis['trade_size_std'] = df['amount'].std()
                analysis['trade_size_median'] = df['amount'].median()
                
                # Large trade detection
                threshold = analysis['avg_trade_size'] + 2 * analysis['trade_size_std']
                large_trades = df[df['amount'] > threshold]
                analysis['large_trade_count'] = len(large_trades)
                analysis['large_trade_volume'] = large_trades['amount'].sum()
            
            # Time-based analysis
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                analysis['trade_frequency'] = len(trades) / 60  # trades per minute
                
                # Recent vs older trades comparison
                cutoff_time = df['timestamp'].max() - timedelta(minutes=5)
                recent_trades = df[df['timestamp'] > cutoff_time]
                
                if len(recent_trades) > 0:
                    analysis['recent_trade_count'] = len(recent_trades)
                    analysis['recent_avg_size'] = recent_trades['amount'].mean()
        
        except Exception as e:
            logger.error(f"Error analyzing trade flow: {e}")
        
        return analysis
    
    async def calculate_order_flow_imbalance(self, symbol: str, connector, lookback_minutes: int = 5) -> Dict[str, Any]:
        """Calculate order flow imbalance over time"""
        imbalance = {}
        
        try:
            # Check cache first
            cache_key = f"{symbol}_{lookback_minutes}"
            if self._is_cache_valid(cache_key):
                return self.imbalance_cache[cache_key]['data']
            
            # Get historical trades
            trades = await connector.fetch_trades(symbol, limit=1000)
            
            if not trades:
                return imbalance
            
            # Filter trades by time
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
            recent_trades = [
                trade for trade in trades 
                if datetime.fromtimestamp(trade['timestamp'] / 1000, tz=timezone.utc) > cutoff_time
            ]
            
            if not recent_trades:
                return imbalance
            
            # Calculate order flow imbalance
            buy_volume = sum(trade['amount'] for trade in recent_trades if trade['side'] == 'buy')
            sell_volume = sum(trade['amount'] for trade in recent_trades if trade['side'] == 'sell')
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                imbalance['order_flow_imbalance'] = (buy_volume - sell_volume) / total_volume
                imbalance['delta_volume'] = buy_volume - sell_volume
                imbalance['cumulative_delta'] = imbalance['delta_volume']  # Would accumulate over time
            
            # Volume-weighted average price of flow
            if recent_trades:
                buy_value = sum(trade['price'] * trade['amount'] for trade in recent_trades if trade['side'] == 'buy')
                sell_value = sum(trade['price'] * trade['amount'] for trade in recent_trades if trade['side'] == 'sell')
                
                if buy_volume > 0:
                    imbalance['avg_buy_price'] = buy_value / buy_volume
                if sell_volume > 0:
                    imbalance['avg_sell_price'] = sell_value / sell_volume
            
            # Cache the result
            self._cache_imbalance(cache_key, imbalance)
            
        except Exception as e:
            logger.error(f"Error calculating order flow imbalance for {symbol}: {e}")
        
        return imbalance
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.imbalance_cache:
            return False
        
        cache_entry = self.imbalance_cache[cache_key]
        cache_age = (datetime.now(timezone.utc) - cache_entry['timestamp']).total_seconds()
        
        return cache_age < self.cache_duration
    
    def _cache_imbalance(self, cache_key: str, data: Dict[str, Any]):
        """Cache imbalance data"""
        self.imbalance_cache[cache_key] = {
            'data': data.copy(),
            'timestamp': datetime.now(timezone.utc)
        }
    
    def get_liquidity_score(self, orderbook: Dict[str, Any], levels: int = 10) -> float:
        """Calculate liquidity score based on order book depth"""
        try:
            bids = orderbook.get('bids', [])[:levels]
            asks = orderbook.get('asks', [])[:levels]
            
            if not bids or not asks:
                return 0.0
            
            # Calculate total volume
            bid_volume = sum(float(bid[1]) for bid in bids)
            ask_volume = sum(float(ask[1]) for ask in asks)
            total_volume = bid_volume + ask_volume
            
            # Calculate spread
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread_percent = ((best_ask - best_bid) / best_bid) * 100
            
            # Score based on volume and tight spread
            volume_score = min(total_volume / 1000, 1.0)  # Normalize to max 1.0
            spread_score = max(0, 1 - spread_percent)  # Lower spread = higher score
            
            # Combined score
            liquidity_score = (volume_score * 0.7 + spread_score * 0.3)
            
            return min(max(liquidity_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating liquidity score: {e}")
            return 0.0
    
    def detect_spoofing_patterns(self, orderbook_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect potential spoofing patterns in order book history"""
        patterns = {}
        
        try:
            if len(orderbook_history) < 10:
                return patterns
            
            # Analyze order placement and cancellation patterns
            large_order_threshold = 10000  # USDT value
            
            for i in range(1, len(orderbook_history)):
                current_book = orderbook_history[i]
                previous_book = orderbook_history[i-1]
                
                # Check for large orders that appear and disappear quickly
                current_bids = {float(bid[0]): float(bid[1]) for bid in current_book.get('bids', [])}
                previous_bids = {float(bid[0]): float(bid[1]) for bid in previous_book.get('bids', [])}
                
                # Find new large orders
                new_large_orders = []
                for price, volume in current_bids.items():
                    if price not in previous_bids and volume * price > large_order_threshold:
                        new_large_orders.append((price, volume))
                
                # Check if these orders disappear in next snapshot
                if new_large_orders and i < len(orderbook_history) - 1:
                    next_book = orderbook_history[i+1]
                    next_bids = {float(bid[0]): float(bid[1]) for bid in next_book.get('bids', [])}
                    
                    disappeared_orders = []
                    for price, volume in new_large_orders:
                        if price not in next_bids:
                            disappeared_orders.append((price, volume))
                    
                    if disappeared_orders:
                        patterns.setdefault('potential_spoofing_events', []).append({
                            'timestamp': i,
                            'disappeared_orders': disappeared_orders
                        })
            
            # Summary statistics
            if 'potential_spoofing_events' in patterns:
                patterns['spoofing_event_count'] = len(patterns['potential_spoofing_events'])
                patterns['spoofing_risk_score'] = min(patterns['spoofing_event_count'] / 10, 1.0)
        
        except Exception as e:
            logger.error(f"Error detecting spoofing patterns: {e}")
        
        return patterns