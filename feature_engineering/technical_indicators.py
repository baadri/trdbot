"""
Technical Indicators Calculator
Calculates various technical analysis indicators
"""

import logging
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    trend: Dict[str, Any]
    momentum: Dict[str, Any]
    volatility: Dict[str, Any]
    volume: Dict[str, Any]

class TechnicalIndicators:
    """Technical indicators calculator"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = self._load_default_config() if config is None else config
        self.cache = {}
    
    def _load_default_config(self) -> Dict:
        """Load default indicator configuration"""
        return {
            'trend': {
                'ema_periods': [9, 21, 55, 200],
                'sma_periods': [20, 50, 200],
                'macd': [12, 26, 9],
                'adx_period': 14,
                'ichimoku': [9, 26, 52, 26]
            },
            'momentum': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'stochastic': [14, 3, 3],
                'williams_r_period': 14,
                'cci_period': 20
            },
            'volatility': {
                'bollinger_bands': [20, 2],
                'atr_period': 14,
                'keltner_channels': [20, 10]
            },
            'volume': {
                'obv_enabled': True,
                'mfi_period': 14,
                'vwap_enabled': True,
                'volume_sma_period': 20
            }
        }
    
    def calculate_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all technical indicators
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary with all calculated indicators
        """
        if df is None or df.empty or len(df) < 50:
            logger.warning("Insufficient data for technical indicators")
            return {}
        
        try:
            indicators = {}
            
            # Calculate trend indicators
            indicators.update(self.calculate_trend_indicators(df))
            
            # Calculate momentum indicators
            indicators.update(self.calculate_momentum_indicators(df))
            
            # Calculate volatility indicators
            indicators.update(self.calculate_volatility_indicators(df))
            
            # Calculate volume indicators
            indicators.update(self.calculate_volume_indicators(df))
            
            # Calculate support/resistance levels
            indicators.update(self.calculate_support_resistance(df))
            
            # Calculate market structure indicators
            indicators.update(self.calculate_market_structure(df))
            
            logger.debug(f"Calculated {len(indicators)} technical indicators")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trend-based indicators"""
        indicators = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Exponential Moving Averages
            for period in self.config['trend']['ema_periods']:
                if len(df) >= period:
                    ema = talib.EMA(close, timeperiod=period)
                    indicators[f'ema_{period}'] = ema[-1] if not np.isnan(ema[-1]) else None
                    
                    # EMA slope (trend direction)
                    if len(ema) >= 5:
                        slope = np.polyfit(range(5), ema[-5:], 1)[0]
                        indicators[f'ema_{period}_slope'] = slope
            
            # Simple Moving Averages
            for period in self.config['trend']['sma_periods']:
                if len(df) >= period:
                    sma = talib.SMA(close, timeperiod=period)
                    indicators[f'sma_{period}'] = sma[-1] if not np.isnan(sma[-1]) else None
            
            # MACD
            fast, slow, signal_period = self.config['trend']['macd']
            if len(df) >= slow + signal_period:
                macd, macd_signal, macd_hist = talib.MACD(
                    close, fastperiod=fast, slowperiod=slow, signalperiod=signal_period
                )
                indicators['macd'] = macd[-1] if not np.isnan(macd[-1]) else None
                indicators['macd_signal'] = macd_signal[-1] if not np.isnan(macd_signal[-1]) else None
                indicators['macd_histogram'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else None
                
                # MACD crossover signals
                if len(macd) >= 2 and len(macd_signal) >= 2:
                    indicators['macd_bullish_crossover'] = (
                        macd[-1] > macd_signal[-1] and macd[-2] <= macd_signal[-2]
                    )
                    indicators['macd_bearish_crossover'] = (
                        macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]
                    )
            
            # ADX (Average Directional Index)
            adx_period = self.config['trend']['adx_period']
            if len(df) >= adx_period * 2:
                adx = talib.ADX(high, low, close, timeperiod=adx_period)
                plus_di = talib.PLUS_DI(high, low, close, timeperiod=adx_period)
                minus_di = talib.MINUS_DI(high, low, close, timeperiod=adx_period)
                
                indicators['adx'] = adx[-1] if not np.isnan(adx[-1]) else None
                indicators['plus_di'] = plus_di[-1] if not np.isnan(plus_di[-1]) else None
                indicators['minus_di'] = minus_di[-1] if not np.isnan(minus_di[-1]) else None
                
                # Trend strength classification
                if indicators['adx'] is not None:
                    if indicators['adx'] > 50:
                        indicators['trend_strength'] = 'very_strong'
                    elif indicators['adx'] > 25:
                        indicators['trend_strength'] = 'strong'
                    elif indicators['adx'] > 20:
                        indicators['trend_strength'] = 'moderate'
                    else:
                        indicators['trend_strength'] = 'weak'
            
            # Ichimoku Cloud
            ichimoku_params = self.config['trend']['ichimoku']
            if len(df) >= max(ichimoku_params):
                indicators.update(self._calculate_ichimoku(df, ichimoku_params))
            
            # Moving Average convergence/divergence analysis
            indicators.update(self._analyze_ma_convergence(df))
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
        
        return indicators
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum-based indicators"""
        indicators = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # RSI
            rsi_period = self.config['momentum']['rsi_period']
            if len(df) >= rsi_period * 2:
                rsi = talib.RSI(close, timeperiod=rsi_period)
                indicators['rsi'] = rsi[-1] if not np.isnan(rsi[-1]) else None
                
                # RSI conditions
                oversold = self.config['momentum']['rsi_oversold']
                overbought = self.config['momentum']['rsi_overbought']
                
                if indicators['rsi'] is not None:
                    indicators['rsi_oversold'] = indicators['rsi'] < oversold
                    indicators['rsi_overbought'] = indicators['rsi'] > overbought
                    
                    # RSI divergences
                    indicators.update(self._detect_rsi_divergence(df, rsi))
            
            # Stochastic Oscillator
            k_period, d_period, smooth_k = self.config['momentum']['stochastic']
            if len(df) >= k_period + d_period:
                slowk, slowd = talib.STOCH(
                    high, low, close,
                    fastk_period=k_period,
                    slowk_period=smooth_k,
                    slowd_period=d_period
                )
                indicators['stoch_k'] = slowk[-1] if not np.isnan(slowk[-1]) else None
                indicators['stoch_d'] = slowd[-1] if not np.isnan(slowd[-1]) else None
                
                # Stochastic signals
                if indicators['stoch_k'] is not None and indicators['stoch_d'] is not None:
                    indicators['stoch_oversold'] = indicators['stoch_k'] < 20
                    indicators['stoch_overbought'] = indicators['stoch_k'] > 80
                    
                    if len(slowk) >= 2 and len(slowd) >= 2:
                        indicators['stoch_bullish_crossover'] = (
                            slowk[-1] > slowd[-1] and slowk[-2] <= slowd[-2]
                        )
                        indicators['stoch_bearish_crossover'] = (
                            slowk[-1] < slowd[-1] and slowk[-2] >= slowd[-2]
                        )
            
            # Williams %R
            williams_period = self.config['momentum']['williams_r_period']
            if len(df) >= williams_period:
                willr = talib.WILLR(high, low, close, timeperiod=williams_period)
                indicators['williams_r'] = willr[-1] if not np.isnan(willr[-1]) else None
                
                if indicators['williams_r'] is not None:
                    indicators['williams_oversold'] = indicators['williams_r'] < -80
                    indicators['williams_overbought'] = indicators['williams_r'] > -20
            
            # CCI (Commodity Channel Index)
            cci_period = self.config['momentum']['cci_period']
            if len(df) >= cci_period:
                cci = talib.CCI(high, low, close, timeperiod=cci_period)
                indicators['cci'] = cci[-1] if not np.isnan(cci[-1]) else None
                
                if indicators['cci'] is not None:
                    indicators['cci_oversold'] = indicators['cci'] < -100
                    indicators['cci_overbought'] = indicators['cci'] > 100
            
            # Rate of Change
            roc_periods = [10, 20, 30]
            for period in roc_periods:
                if len(df) >= period:
                    roc = talib.ROC(close, timeperiod=period)
                    indicators[f'roc_{period}'] = roc[-1] if not np.isnan(roc[-1]) else None
            
            # Money Flow Index
            if 'volume' in df.columns and len(df) >= 14:
                volume = df['volume'].values
                mfi = talib.MFI(high, low, close, volume, timeperiod=14)
                indicators['mfi'] = mfi[-1] if not np.isnan(mfi[-1]) else None
                
                if indicators['mfi'] is not None:
                    indicators['mfi_oversold'] = indicators['mfi'] < 20
                    indicators['mfi_overbought'] = indicators['mfi'] > 80
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
        
        return indicators
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility-based indicators"""
        indicators = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Bollinger Bands
            bb_period, bb_std = self.config['volatility']['bollinger_bands']
            if len(df) >= bb_period:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std
                )
                
                indicators['bb_upper'] = bb_upper[-1] if not np.isnan(bb_upper[-1]) else None
                indicators['bb_middle'] = bb_middle[-1] if not np.isnan(bb_middle[-1]) else None
                indicators['bb_lower'] = bb_lower[-1] if not np.isnan(bb_lower[-1]) else None
                
                # Bollinger Band position
                if all(x is not None for x in [indicators['bb_upper'], indicators['bb_lower']]):
                    bb_position = (close[-1] - indicators['bb_lower']) / (
                        indicators['bb_upper'] - indicators['bb_lower']
                    )
                    indicators['bb_position'] = bb_position
                    
                    # Bollinger Band squeeze
                    if len(bb_upper) >= 20:
                        current_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
                        avg_width = np.mean((bb_upper[-20:-1] - bb_lower[-20:-1]) / bb_middle[-20:-1])
                        indicators['bb_squeeze'] = current_width < avg_width * 0.8
            
            # Average True Range
            atr_period = self.config['volatility']['atr_period']
            if len(df) >= atr_period:
                atr = talib.ATR(high, low, close, timeperiod=atr_period)
                indicators['atr'] = atr[-1] if not np.isnan(atr[-1]) else None
                
                # ATR percentage
                if indicators['atr'] is not None:
                    indicators['atr_percent'] = (indicators['atr'] / close[-1]) * 100
            
            # Keltner Channels
            kc_period, kc_multiplier = self.config['volatility']['keltner_channels']
            if len(df) >= kc_period and indicators.get('atr') is not None:
                ema = talib.EMA(close, timeperiod=kc_period)
                if not np.isnan(ema[-1]):
                    kc_middle = ema[-1]
                    kc_upper = kc_middle + (indicators['atr'] * kc_multiplier)
                    kc_lower = kc_middle - (indicators['atr'] * kc_multiplier)
                    
                    indicators['kc_upper'] = kc_upper
                    indicators['kc_middle'] = kc_middle
                    indicators['kc_lower'] = kc_lower
                    
                    # Keltner Channel position
                    kc_position = (close[-1] - kc_lower) / (kc_upper - kc_lower)
                    indicators['kc_position'] = kc_position
            
            # Historical Volatility
            if len(df) >= 30:
                returns = np.log(close[1:] / close[:-1])
                hist_vol = np.std(returns[-30:]) * np.sqrt(252) * 100  # Annualized
                indicators['historical_volatility'] = hist_vol
            
            # Volatility ratio
            if len(df) >= 20:
                short_vol = np.std(np.log(close[-10:] / close[-11:-1])) * np.sqrt(252)
                long_vol = np.std(np.log(close[-20:] / close[-21:-1])) * np.sqrt(252)
                if long_vol != 0:
                    indicators['volatility_ratio'] = short_vol / long_vol
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
        
        return indicators
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based indicators"""
        indicators = {}
        
        if 'volume' not in df.columns:
            logger.warning("Volume data not available")
            return indicators
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # On-Balance Volume
            if self.config['volume']['obv_enabled']:
                obv = talib.OBV(close, volume)
                indicators['obv'] = obv[-1] if not np.isnan(obv[-1]) else None
                
                # OBV trend
                if len(obv) >= 10:
                    obv_slope = np.polyfit(range(10), obv[-10:], 1)[0]
                    indicators['obv_trend'] = 'bullish' if obv_slope > 0 else 'bearish'
            
            # Volume SMA
            vol_sma_period = self.config['volume']['volume_sma_period']
            if len(df) >= vol_sma_period:
                vol_sma = talib.SMA(volume, timeperiod=vol_sma_period)
                indicators['volume_sma'] = vol_sma[-1] if not np.isnan(vol_sma[-1]) else None
                
                # Volume ratio
                if indicators['volume_sma'] is not None and indicators['volume_sma'] > 0:
                    indicators['volume_ratio'] = volume[-1] / indicators['volume_sma']
            
            # VWAP (Volume Weighted Average Price)
            if self.config['volume']['vwap_enabled']:
                vwap = self._calculate_vwap(df)
                if vwap is not None:
                    indicators['vwap'] = vwap
                    indicators['price_vs_vwap'] = (close[-1] - vwap) / vwap
            
            # Accumulation/Distribution Line
            ad = talib.AD(high, low, close, volume)
            indicators['ad_line'] = ad[-1] if not np.isnan(ad[-1]) else None
            
            # Chaikin Money Flow
            if len(df) >= 20:
                cmf = self._calculate_cmf(df, 20)
                indicators['cmf'] = cmf
            
            # Volume Price Trend
            vpt = self._calculate_vpt(df)
            if len(vpt) > 0:
                indicators['vpt'] = vpt[-1]
                
                # VPT trend
                if len(vpt) >= 10:
                    vpt_slope = np.polyfit(range(10), vpt[-10:], 1)[0]
                    indicators['vpt_trend'] = 'bullish' if vpt_slope > 0 else 'bearish'
            
            # Volume oscillator
            if len(df) >= 28:
                vol_osc = self._calculate_volume_oscillator(volume, 14, 28)
                indicators['volume_oscillator'] = vol_osc
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
        
        return indicators
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        indicators = {}
        
        try:
            if len(df) < window * 2:
                return indicators
            
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Pivot points
            pivot_points = self._calculate_pivot_points(df.iloc[-1])
            indicators.update(pivot_points)
            
            # Dynamic support/resistance using swing highs/lows
            swing_highs = self._find_swing_points(high, window, 'high')
            swing_lows = self._find_swing_points(low, window, 'low')
            
            # Recent significant levels
            recent_highs = [high[i] for i in swing_highs if i >= len(high) - window * 3]
            recent_lows = [low[i] for i in swing_lows if i >= len(low) - window * 3]
            
            if recent_highs:
                indicators['resistance_level'] = max(recent_highs)
                indicators['resistance_strength'] = len([h for h in recent_highs if abs(h - indicators['resistance_level']) < indicators['resistance_level'] * 0.005])
            
            if recent_lows:
                indicators['support_level'] = min(recent_lows)
                indicators['support_strength'] = len([l for l in recent_lows if abs(l - indicators['support_level']) < indicators['support_level'] * 0.005])
            
            # Distance to support/resistance
            current_price = close[-1]
            if 'resistance_level' in indicators:
                indicators['distance_to_resistance'] = (indicators['resistance_level'] - current_price) / current_price
            
            if 'support_level' in indicators:
                indicators['distance_to_support'] = (current_price - indicators['support_level']) / current_price
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
        
        return indicators
    
    def calculate_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market structure indicators"""
        indicators = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Higher highs and higher lows (uptrend)
            # Lower highs and lower lows (downtrend)
            if len(df) >= 20:
                recent_highs = []
                recent_lows = []
                
                for i in range(len(high) - 20, len(high)):
                    if i > 2 and i < len(high) - 2:
                        if high[i] > high[i-1] and high[i] > high[i+1]:
                            recent_highs.append(high[i])
                        if low[i] < low[i-1] and low[i] < low[i+1]:
                            recent_lows.append(low[i])
                
                if len(recent_highs) >= 2:
                    indicators['higher_highs'] = recent_highs[-1] > recent_highs[-2]
                
                if len(recent_lows) >= 2:
                    indicators['higher_lows'] = recent_lows[-1] > recent_lows[-2]
                
                # Market structure trend
                if indicators.get('higher_highs') and indicators.get('higher_lows'):
                    indicators['market_structure'] = 'uptrend'
                elif not indicators.get('higher_highs', True) and not indicators.get('higher_lows', True):
                    indicators['market_structure'] = 'downtrend'
                else:
                    indicators['market_structure'] = 'sideways'
            
            # Price position in recent range
            if len(df) >= 50:
                recent_high = np.max(high[-50:])
                recent_low = np.min(low[-50:])
                current_price = close[-1]
                
                if recent_high != recent_low:
                    price_position = (current_price - recent_low) / (recent_high - recent_low)
                    indicators['price_position_in_range'] = price_position
            
        except Exception as e:
            logger.error(f"Error calculating market structure: {e}")
        
        return indicators
    
    def _calculate_ichimoku(self, df: pd.DataFrame, params: List[int]) -> Dict[str, Any]:
        """Calculate Ichimoku Cloud indicators"""
        indicators = {}
        
        try:
            conversion_line_period, base_line_period, leading_span_b_period, lagging_span_period = params
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Conversion Line (Tenkan-sen)
            if len(df) >= conversion_line_period:
                conversion_high = np.max(high[-conversion_line_period:])
                conversion_low = np.min(low[-conversion_line_period:])
                indicators['tenkan_sen'] = (conversion_high + conversion_low) / 2
            
            # Base Line (Kijun-sen)
            if len(df) >= base_line_period:
                base_high = np.max(high[-base_line_period:])
                base_low = np.min(low[-base_line_period:])
                indicators['kijun_sen'] = (base_high + base_low) / 2
            
            # Leading Span A (Senkou Span A)
            if 'tenkan_sen' in indicators and 'kijun_sen' in indicators:
                indicators['senkou_span_a'] = (indicators['tenkan_sen'] + indicators['kijun_sen']) / 2
            
            # Leading Span B (Senkou Span B)
            if len(df) >= leading_span_b_period:
                span_b_high = np.max(high[-leading_span_b_period:])
                span_b_low = np.min(low[-leading_span_b_period:])
                indicators['senkou_span_b'] = (span_b_high + span_b_low) / 2
            
            # Cloud analysis
            if 'senkou_span_a' in indicators and 'senkou_span_b' in indicators:
                current_price = close[-1]
                span_a = indicators['senkou_span_a']
                span_b = indicators['senkou_span_b']
                
                # Price vs Cloud
                cloud_top = max(span_a, span_b)
                cloud_bottom = min(span_a, span_b)
                
                if current_price > cloud_top:
                    indicators['ichimoku_signal'] = 'bullish'
                elif current_price < cloud_bottom:
                    indicators['ichimoku_signal'] = 'bearish'
                else:
                    indicators['ichimoku_signal'] = 'neutral'
                
                # Cloud color
                indicators['cloud_color'] = 'green' if span_a > span_b else 'red'
        
        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {e}")
        
        return indicators
    
    def _analyze_ma_convergence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze moving average convergence/divergence"""
        indicators = {}
        
        try:
            close = df['close'].values
            
            if len(df) >= 200:
                ema_9 = talib.EMA(close, timeperiod=9)
                ema_21 = talib.EMA(close, timeperiod=21)
                ema_55 = talib.EMA(close, timeperiod=55)
                sma_200 = talib.SMA(close, timeperiod=200)
                
                # Golden Cross / Death Cross
                if len(ema_21) >= 2 and len(sma_200) >= 2:
                    indicators['golden_cross'] = (
                        ema_21[-1] > sma_200[-1] and ema_21[-2] <= sma_200[-2]
                    )
                    indicators['death_cross'] = (
                        ema_21[-1] < sma_200[-1] and ema_21[-2] >= sma_200[-2]
                    )
                
                # EMA alignment
                if all(not np.isnan(x[-1]) for x in [ema_9, ema_21, ema_55]):
                    indicators['ema_bullish_alignment'] = (
                        ema_9[-1] > ema_21[-1] > ema_55[-1]
                    )
                    indicators['ema_bearish_alignment'] = (
                        ema_9[-1] < ema_21[-1] < ema_55[-1]
                    )
        
        except Exception as e:
            logger.error(f"Error analyzing MA convergence: {e}")
        
        return indicators
    
    def _detect_rsi_divergence(self, df: pd.DataFrame, rsi: np.ndarray) -> Dict[str, Any]:
        """Detect RSI divergences"""
        indicators = {}
        
        try:
            if len(df) < 20 or len(rsi) < 20:
                return indicators
            
            close = df['close'].values
            
            # Find recent swing points
            price_highs = self._find_swing_points(close[-20:], 5, 'high')
            price_lows = self._find_swing_points(close[-20:], 5, 'low')
            rsi_highs = self._find_swing_points(rsi[-20:], 5, 'high')
            rsi_lows = self._find_swing_points(rsi[-20:], 5, 'low')
            
            # Bullish divergence (price makes lower low, RSI makes higher low)
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                last_price_low = price_lows[-1]
                prev_price_low = price_lows[-2]
                last_rsi_low = rsi_lows[-1]
                prev_rsi_low = rsi_lows[-2]
                
                if (close[last_price_low] < close[prev_price_low] and 
                    rsi[last_rsi_low] > rsi[prev_rsi_low]):
                    indicators['rsi_bullish_divergence'] = True
            
            # Bearish divergence (price makes higher high, RSI makes lower high)
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                last_price_high = price_highs[-1]
                prev_price_high = price_highs[-2]
                last_rsi_high = rsi_highs[-1]
                prev_rsi_high = rsi_highs[-2]
                
                if (close[last_price_high] > close[prev_price_high] and 
                    rsi[last_rsi_high] < rsi[prev_rsi_high]):
                    indicators['rsi_bearish_divergence'] = True
        
        except Exception as e:
            logger.error(f"Error detecting RSI divergence: {e}")
        
        return indicators
    
    def _calculate_vwap(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = np.sum(typical_price * df['volume']) / np.sum(df['volume'])
            return float(vwap)
        except Exception:
            return None
    
    def _calculate_cmf(self, df: pd.DataFrame, period: int) -> Optional[float]:
        """Calculate Chaikin Money Flow"""
        try:
            if len(df) < period:
                return None
            
            mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mf_volume = mf_multiplier * df['volume']
            
            cmf = mf_volume.rolling(period).sum() / df['volume'].rolling(period).sum()
            return float(cmf.iloc[-1])
        except Exception:
            return None
    
    def _calculate_vpt(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate Volume Price Trend"""
        try:
            price_change = df['close'].pct_change()
            vpt = (price_change * df['volume']).cumsum()
            return vpt.values
        except Exception:
            return np.array([])
    
    def _calculate_volume_oscillator(self, volume: np.ndarray, short_period: int, long_period: int) -> Optional[float]:
        """Calculate Volume Oscillator"""
        try:
            short_ma = talib.SMA(volume, timeperiod=short_period)
            long_ma = talib.SMA(volume, timeperiod=long_period)
            
            if not np.isnan(short_ma[-1]) and not np.isnan(long_ma[-1]) and long_ma[-1] != 0:
                vol_osc = ((short_ma[-1] - long_ma[-1]) / long_ma[-1]) * 100
                return float(vol_osc)
        except Exception:
            pass
        
        return None
    
    def _calculate_pivot_points(self, last_candle: pd.Series) -> Dict[str, float]:
        """Calculate pivot points"""
        try:
            high = last_candle['high']
            low = last_candle['low']
            close = last_candle['close']
            
            pivot = (high + low + close) / 3
            
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot_point': pivot,
                'resistance_1': r1,
                'resistance_2': r2,
                'resistance_3': r3,
                'support_1': s1,
                'support_2': s2,
                'support_3': s3
            }
        except Exception:
            return {}
    
    def _find_swing_points(self, data: np.ndarray, window: int, point_type: str) -> List[int]:
        """Find swing high/low points"""
        swing_points = []
        
        try:
            for i in range(window, len(data) - window):
                if point_type == 'high':
                    if all(data[i] >= data[i-j] for j in range(1, window+1)) and \
                       all(data[i] >= data[i+j] for j in range(1, window+1)):
                        swing_points.append(i)
                elif point_type == 'low':
                    if all(data[i] <= data[i-j] for j in range(1, window+1)) and \
                       all(data[i] <= data[i+j] for j in range(1, window+1)):
                        swing_points.append(i)
        except Exception:
            pass
        
        return swing_points