"""
Sentiment Scraper
Collects and analyzes market sentiment from various sources
"""

import logging
import asyncio
import aiohttp
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from dataclasses import dataclass
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    """Sentiment data structure"""
    source: str
    text: str
    sentiment_score: float  # -1 to 1
    confidence: float
    timestamp: datetime
    symbol: Optional[str] = None
    author: Optional[str] = None
    engagement: Optional[int] = None

class SentimentScraper:
    """Market sentiment scraper and analyzer"""
    
    def __init__(self):
        self.session = None
        self.sentiment_cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Cryptocurrency-related keywords
        self.crypto_keywords = {
            'BTC': ['bitcoin', 'btc', '$btc', '#btc', '#bitcoin'],
            'ETH': ['ethereum', 'eth', '$eth', '#eth', '#ethereum'],
            'BNB': ['binance', 'bnb', '$bnb', '#bnb'],
            'SOL': ['solana', 'sol', '$sol', '#sol', '#solana'],
            'XRP': ['ripple', 'xrp', '$xrp', '#xrp', '#ripple'],
            'ADA': ['cardano', 'ada', '$ada', '#ada', '#cardano'],
            'AVAX': ['avalanche', 'avax', '$avax', '#avax', '#avalanche'],
            'DOT': ['polkadot', 'dot', '$dot', '#dot', '#polkadot'],
            'MATIC': ['polygon', 'matic', '$matic', '#matic', '#polygon'],
            'LINK': ['chainlink', 'link', '$link', '#link', '#chainlink']
        }
        
        # Sentiment keywords
        self.positive_keywords = [
            'bullish', 'moon', 'pump', 'rally', 'breakout', 'surge', 'rocket',
            'buy', 'long', 'hodl', 'accumulate', 'gem', 'bullrun', 'to the moon',
            'diamond hands', 'strong', 'support', 'bounce', 'recovery', 'green'
        ]
        
        self.negative_keywords = [
            'bearish', 'dump', 'crash', 'dip', 'fall', 'sell', 'short',
            'liquidated', 'rekt', 'panic', 'fear', 'correction', 'resistance',
            'breakdown', 'red', 'bearmarket', 'capitulation', 'weak', 'drop'
        ]
        
        # API endpoints and headers
        self.reddit_headers = {
            'User-Agent': 'CryptoTradingBot/1.0 (by /u/cryptotrader)'
        }
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """
        Get overall market sentiment from multiple sources
        
        Returns:
            Dictionary with sentiment analysis results
        """
        cache_key = "market_sentiment"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            logger.debug("Returning cached market sentiment")
            return self.sentiment_cache[cache_key]['data']
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            sentiment_data = {}
            
            # Collect from different sources
            reddit_sentiment = await self._get_reddit_sentiment()
            if reddit_sentiment:
                sentiment_data['reddit'] = reddit_sentiment
            
            # Twitter/X sentiment (using alternative methods due to API restrictions)
            twitter_sentiment = await self._get_twitter_sentiment()
            if twitter_sentiment:
                sentiment_data['twitter'] = twitter_sentiment
            
            # Fear & Greed Index
            fear_greed = await self._get_fear_greed_index()
            if fear_greed:
                sentiment_data['fear_greed'] = fear_greed
            
            # News sentiment
            news_sentiment = await self._get_news_sentiment()
            if news_sentiment:
                sentiment_data['news'] = news_sentiment
            
            # Aggregate sentiment scores
            aggregated = self._aggregate_sentiment(sentiment_data)
            sentiment_data['aggregated'] = aggregated
            
            # Cache results
            self._cache_sentiment(cache_key, sentiment_data)
            
            logger.info("Market sentiment analysis completed")
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return {}
    
    async def get_symbol_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get sentiment for a specific cryptocurrency symbol
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            Dictionary with symbol-specific sentiment
        """
        cache_key = f"symbol_sentiment_{symbol}"
        
        if self._is_cache_valid(cache_key):
            return self.sentiment_cache[cache_key]['data']
        
        try:
            symbol_sentiment = {}
            
            # Get symbol-specific keywords
            keywords = self.crypto_keywords.get(symbol, [symbol.lower()])
            
            # Reddit analysis for specific symbol
            reddit_data = await self._get_reddit_symbol_sentiment(symbol, keywords)
            if reddit_data:
                symbol_sentiment['reddit'] = reddit_data
            
            # News analysis for specific symbol
            news_data = await self._get_news_symbol_sentiment(symbol, keywords)
            if news_data:
                symbol_sentiment['news'] = news_data
            
            # Social media mentions analysis
            mentions_data = await self._analyze_social_mentions(symbol, keywords)
            if mentions_data:
                symbol_sentiment['social_mentions'] = mentions_data
            
            # Aggregate symbol sentiment
            aggregated = self._aggregate_symbol_sentiment(symbol_sentiment)
            symbol_sentiment['aggregated'] = aggregated
            
            self._cache_sentiment(cache_key, symbol_sentiment)
            
            return symbol_sentiment
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")
            return {}
    
    async def _get_reddit_sentiment(self) -> Optional[Dict[str, Any]]:
        """Get sentiment from Reddit crypto communities"""
        try:
            subreddits = ['cryptocurrency', 'bitcoin', 'ethtrader', 'cryptomarkets']
            all_posts = []
            
            for subreddit in subreddits:
                posts = await self._fetch_reddit_posts(subreddit, limit=25)
                if posts:
                    all_posts.extend(posts)
            
            if not all_posts:
                return None
            
            # Analyze sentiment of posts
            sentiment_scores = []
            for post in all_posts:
                score = self._analyze_text_sentiment(post['title'] + ' ' + post.get('selftext', ''))
                sentiment_scores.append(score)
            
            # Calculate aggregated metrics
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            positive_ratio = len([s for s in sentiment_scores if s > 0.1]) / len(sentiment_scores)
            negative_ratio = len([s for s in sentiment_scores if s < -0.1]) / len(sentiment_scores)
            
            return {
                'avg_sentiment': avg_sentiment,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'neutral_ratio': 1 - positive_ratio - negative_ratio,
                'post_count': len(all_posts),
                'confidence': min(len(all_posts) / 100, 1.0)  # Higher confidence with more posts
            }
            
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {e}")
            return None
    
    async def _fetch_reddit_posts(self, subreddit: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Fetch posts from a Reddit subreddit"""
        try:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
            
            async with self.session.get(url, headers=self.reddit_headers) as response:
                if response.status == 200:
                    data = await response.json()
                    posts = []
                    
                    for post_data in data['data']['children']:
                        post = post_data['data']
                        posts.append({
                            'title': post.get('title', ''),
                            'selftext': post.get('selftext', ''),
                            'score': post.get('score', 0),
                            'upvote_ratio': post.get('upvote_ratio', 0.5),
                            'num_comments': post.get('num_comments', 0),
                            'created_utc': post.get('created_utc', 0)
                        })
                    
                    return posts
                
        except Exception as e:
            logger.error(f"Error fetching Reddit posts from {subreddit}: {e}")
        
        return []
    
    async def _get_twitter_sentiment(self) -> Optional[Dict[str, Any]]:
        """Get sentiment from Twitter/X (using alternative methods)"""
        try:
            # Since Twitter API requires authentication, we'll use alternative methods
            # This could be replaced with actual Twitter API integration
            
            # For now, return a placeholder that could be populated with real data
            # In a production environment, you would:
            # 1. Use Twitter API v2 with proper authentication
            # 2. Search for crypto-related tweets
            # 3. Analyze sentiment of tweet content
            
            # Placeholder implementation
            return {
                'avg_sentiment': 0.0,
                'positive_ratio': 0.4,
                'negative_ratio': 0.3,
                'neutral_ratio': 0.3,
                'tweet_count': 0,
                'confidence': 0.0,
                'note': 'Twitter API integration required'
            }
            
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment: {e}")
            return None
    
    async def _get_fear_greed_index(self) -> Optional[Dict[str, Any]]:
        """Get Fear & Greed Index from alternative-me.com"""
        try:
            url = "https://api.alternative.me/fng/"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('data'):
                        current_data = data['data'][0]
                        
                        return {
                            'value': int(current_data.get('value', 50)),
                            'classification': current_data.get('value_classification', 'Neutral'),
                            'timestamp': current_data.get('timestamp'),
                            'sentiment_score': self._fear_greed_to_sentiment(int(current_data.get('value', 50)))
                        }
                
        except Exception as e:
            logger.error(f"Error getting Fear & Greed Index: {e}")
        
        return None
    
    def _fear_greed_to_sentiment(self, fg_value: int) -> float:
        """Convert Fear & Greed Index to sentiment score (-1 to 1)"""
        # Fear & Greed Index: 0 (Extreme Fear) to 100 (Extreme Greed)
        # Sentiment score: -1 (Very Negative) to 1 (Very Positive)
        return (fg_value - 50) / 50
    
    async def _get_news_sentiment(self) -> Optional[Dict[str, Any]]:
        """Get sentiment from cryptocurrency news"""
        try:
            # Using CoinDesk RSS feed as an example
            news_sources = [
                'https://feeds.feedburner.com/CoinDesk',
                # Add more news sources as needed
            ]
            
            all_articles = []
            
            for source in news_sources:
                articles = await self._fetch_news_articles(source)
                if articles:
                    all_articles.extend(articles)
            
            if not all_articles:
                return None
            
            # Analyze sentiment of headlines and descriptions
            sentiment_scores = []
            for article in all_articles:
                text = article.get('title', '') + ' ' + article.get('description', '')
                score = self._analyze_text_sentiment(text)
                sentiment_scores.append(score)
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            positive_ratio = len([s for s in sentiment_scores if s > 0.1]) / len(sentiment_scores)
            negative_ratio = len([s for s in sentiment_scores if s < -0.1]) / len(sentiment_scores)
            
            return {
                'avg_sentiment': avg_sentiment,
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'neutral_ratio': 1 - positive_ratio - negative_ratio,
                'article_count': len(all_articles),
                'confidence': min(len(all_articles) / 50, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error getting news sentiment: {e}")
            return None
    
    async def _fetch_news_articles(self, rss_url: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch articles from RSS feed"""
        try:
            async with self.session.get(rss_url) as response:
                if response.status == 200:
                    # For now, return empty list
                    # In production, you would parse the RSS feed
                    # and extract article titles, descriptions, and timestamps
                    return []
                
        except Exception as e:
            logger.error(f"Error fetching news from {rss_url}: {e}")
        
        return []
    
    async def _get_reddit_symbol_sentiment(self, symbol: str, keywords: List[str]) -> Optional[Dict[str, Any]]:
        """Get Reddit sentiment for a specific symbol"""
        try:
            # Search for posts mentioning the symbol
            search_terms = ' OR '.join(keywords[:3])  # Limit search terms
            subreddits = ['cryptocurrency', 'bitcoin', 'ethtrader']
            
            all_posts = []
            for subreddit in subreddits:
                # In production, you would use Reddit's search API
                # For now, we'll use the hot posts and filter by keywords
                posts = await self._fetch_reddit_posts(subreddit, limit=50)
                
                # Filter posts containing our keywords
                filtered_posts = []
                for post in posts:
                    text = (post.get('title', '') + ' ' + post.get('selftext', '')).lower()
                    if any(keyword.lower() in text for keyword in keywords):
                        filtered_posts.append(post)
                
                all_posts.extend(filtered_posts)
            
            if not all_posts:
                return None
            
            # Analyze sentiment
            sentiment_scores = []
            for post in all_posts:
                text = post['title'] + ' ' + post.get('selftext', '')
                score = self._analyze_text_sentiment(text)
                sentiment_scores.append(score)
            
            return {
                'avg_sentiment': sum(sentiment_scores) / len(sentiment_scores),
                'post_count': len(all_posts),
                'positive_mentions': len([s for s in sentiment_scores if s > 0.1]),
                'negative_mentions': len([s for s in sentiment_scores if s < -0.1]),
                'confidence': min(len(all_posts) / 20, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment for {symbol}: {e}")
            return None
    
    async def _get_news_symbol_sentiment(self, symbol: str, keywords: List[str]) -> Optional[Dict[str, Any]]:
        """Get news sentiment for a specific symbol"""
        try:
            # This would search news sources for articles mentioning the symbol
            # For now, return a placeholder
            return {
                'avg_sentiment': 0.0,
                'article_count': 0,
                'positive_articles': 0,
                'negative_articles': 0,
                'confidence': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return None
    
    async def _analyze_social_mentions(self, symbol: str, keywords: List[str]) -> Optional[Dict[str, Any]]:
        """Analyze social media mentions for a symbol"""
        try:
            # This would analyze mentions across various social platforms
            # For now, return a placeholder
            return {
                'mention_count': 0,
                'sentiment_trend': 'neutral',
                'engagement_score': 0.0,
                'confidence': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing social mentions for {symbol}: {e}")
            return None
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using keyword-based approach
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score from -1 (very negative) to 1 (very positive)
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Count positive and negative keywords
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
        
        # Calculate score
        total_keywords = positive_count + negative_count
        
        if total_keywords == 0:
            return 0.0  # Neutral
        
        sentiment_score = (positive_count - negative_count) / total_keywords
        
        # Apply word weighting (longer texts might have diluted sentiment)
        word_count = len(text.split())
        if word_count > 0:
            keyword_density = total_keywords / word_count
            sentiment_score *= min(keyword_density * 10, 1.0)  # Cap the multiplier
        
        return max(-1.0, min(1.0, sentiment_score))
    
    def _aggregate_sentiment(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate sentiment from multiple sources"""
        try:
            sources = []
            weights = {
                'reddit': 0.3,
                'twitter': 0.25,
                'news': 0.25,
                'fear_greed': 0.2
            }
            
            weighted_sentiment = 0.0
            total_weight = 0.0
            confidence_sum = 0.0
            
            for source, weight in weights.items():
                if source in sentiment_data:
                    data = sentiment_data[source]
                    
                    if source == 'fear_greed':
                        sentiment = data.get('sentiment_score', 0.0)
                        confidence = 1.0  # Fear & Greed Index is always available
                    else:
                        sentiment = data.get('avg_sentiment', 0.0)
                        confidence = data.get('confidence', 0.0)
                    
                    weighted_sentiment += sentiment * weight * confidence
                    total_weight += weight * confidence
                    confidence_sum += confidence
                    sources.append(source)
            
            if total_weight > 0:
                final_sentiment = weighted_sentiment / total_weight
                avg_confidence = confidence_sum / len(sources) if sources else 0.0
            else:
                final_sentiment = 0.0
                avg_confidence = 0.0
            
            # Classify sentiment
            if final_sentiment > 0.3:
                classification = 'very_positive'
            elif final_sentiment > 0.1:
                classification = 'positive'
            elif final_sentiment > -0.1:
                classification = 'neutral'
            elif final_sentiment > -0.3:
                classification = 'negative'
            else:
                classification = 'very_negative'
            
            return {
                'sentiment_score': final_sentiment,
                'classification': classification,
                'confidence': avg_confidence,
                'sources_used': sources,
                'source_count': len(sources)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating sentiment: {e}")
            return {
                'sentiment_score': 0.0,
                'classification': 'neutral',
                'confidence': 0.0,
                'sources_used': [],
                'source_count': 0
            }
    
    def _aggregate_symbol_sentiment(self, symbol_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate sentiment for a specific symbol"""
        try:
            sentiments = []
            confidences = []
            
            for source_data in symbol_sentiment.values():
                if isinstance(source_data, dict) and 'avg_sentiment' in source_data:
                    sentiments.append(source_data['avg_sentiment'])
                    confidences.append(source_data.get('confidence', 0.0))
            
            if not sentiments:
                return {
                    'sentiment_score': 0.0,
                    'classification': 'neutral',
                    'confidence': 0.0
                }
            
            # Weighted average
            weighted_sentiment = sum(s * c for s, c in zip(sentiments, confidences))
            total_confidence = sum(confidences)
            
            if total_confidence > 0:
                final_sentiment = weighted_sentiment / total_confidence
                avg_confidence = total_confidence / len(confidences)
            else:
                final_sentiment = sum(sentiments) / len(sentiments)
                avg_confidence = 0.0
            
            # Classify
            if final_sentiment > 0.2:
                classification = 'positive'
            elif final_sentiment < -0.2:
                classification = 'negative'
            else:
                classification = 'neutral'
            
            return {
                'sentiment_score': final_sentiment,
                'classification': classification,
                'confidence': avg_confidence
            }
            
        except Exception as e:
            logger.error(f"Error aggregating symbol sentiment: {e}")
            return {
                'sentiment_score': 0.0,
                'classification': 'neutral',
                'confidence': 0.0
            }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached sentiment data is still valid"""
        if cache_key not in self.sentiment_cache:
            return False
        
        cache_entry = self.sentiment_cache[cache_key]
        cache_age = (datetime.now(timezone.utc) - cache_entry['timestamp']).total_seconds()
        
        return cache_age < self.cache_duration
    
    def _cache_sentiment(self, cache_key: str, data: Dict[str, Any]):
        """Cache sentiment data"""
        self.sentiment_cache[cache_key] = {
            'data': data.copy(),
            'timestamp': datetime.now(timezone.utc)
        }
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
    
    def get_sentiment_summary(self, sentiment_data: Dict[str, Any]) -> str:
        """Get a human-readable sentiment summary"""
        try:
            if 'aggregated' not in sentiment_data:
                return "No sentiment data available"
            
            agg = sentiment_data['aggregated']
            score = agg.get('sentiment_score', 0.0)
            classification = agg.get('classification', 'neutral')
            confidence = agg.get('confidence', 0.0)
            sources = agg.get('source_count', 0)
            
            summary = f"Market sentiment: {classification.replace('_', ' ').title()} "
            summary += f"(Score: {score:.2f}, Confidence: {confidence:.0%})"
            
            if sources > 0:
                summary += f" - Based on {sources} sources"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating sentiment summary: {e}")
            return "Error generating sentiment summary"