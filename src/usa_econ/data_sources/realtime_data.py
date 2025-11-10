from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import requests
import json
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

try:
    import newspaper
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False


class RealTimeDataManager:
    """Manager for real-time and alternative economic data sources."""
    
    def __init__(self, config: Dict[str, str] = None):
        """Initialize the real-time data manager."""
        self.config = config or {}
        self.session = requests.Session()
        
        # API endpoints
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.news_api_url = "https://newsapi.org/v2/everything"
        
        # Rate limiting
        self.last_request_time = {}
        self.rate_limits = {
            'alpha_vantage': 5,  # 5 requests per minute
            'news_api': 1000,    # 1000 requests per day
            'twitter': 300       # 300 requests per 15 minutes
        }
    
    def _rate_limit_check(self, api_name: str):
        """Check and enforce rate limiting."""
        if api_name in self.last_request_time:
            time_since_last = time.time() - self.last_request_time[api_name]
            min_interval = 60 / self.rate_limits.get(api_name, 60)
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)
        
        self.last_request_time[api_name] = time.time()
    
    def get_market_data_yahoo(
        self,
        symbols: List[str],
        period: str = "1mo",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Get real-time market data from Yahoo Finance.
        
        Args:
            symbols: List of ticker symbols
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with market data
        """
        
        if not YFINANCE_AVAILABLE:
            raise ImportError("Install yfinance: pip install yfinance")
        
        data_dict = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)
                
                if not hist.empty:
                    # Get additional info
                    info = ticker.info
                    data_dict[symbol] = hist
                    
                    # Add fundamental data
                    if 'marketCap' in info:
                        data_dict[f'{symbol}_market_cap'] = info['marketCap']
                    if 'forwardPE' in info:
                        data_dict[f'{symbol}_pe_ratio'] = info['forwardPE']
                    if 'dividendYield' in info:
                        data_dict[f'{symbol}_dividend_yield'] = info['dividendYield']
                        
            except Exception as e:
                print(f"Warning: Could not fetch data for {symbol}: {e}")
        
        return pd.concat(data_dict.values(), axis=1, keys=data_dict.keys())
    
    def get_economic_calendar(
        self,
        days_ahead: int = 7,
        importance: str = "high"
    ) -> pd.DataFrame:
        """Get upcoming economic calendar events.
        
        Args:
            days_ahead: Number of days ahead to fetch
            importance: Event importance (high, medium, low, all)
            
        Returns:
            DataFrame with economic calendar
        """
        
        # Using Forex Factory API (free, no key required)
        url = "https://www.forexfactory.com/calendar.xml"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse XML response (simplified)
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            events = []
            current_date = datetime.now()
            end_date = current_date + timedelta(days=days_ahead)
            
            for event in root.findall('.//event'):
                event_date = datetime.strptime(event.find('date').text, '%Y-%m-%d')
                
                if current_date <= event_date <= end_date:
                    importance_level = event.find('impact').text.lower()
                    
                    if importance == "all" or importance_level == importance:
                        events.append({
                            'date': event_date,
                            'time': event.find('time').text,
                            'currency': event.find('currency').text,
                            'event': event.find('title').text,
                            'importance': importance_level,
                            'forecast': event.find('forecast').text,
                            'previous': event.find('previous').text
                        })
            
            return pd.DataFrame(events)
            
        except Exception as e:
            print(f"Warning: Could not fetch economic calendar: {e}")
            return pd.DataFrame()
    
    def get_market_sentiment(
        self,
        sources: List[str] = ["fear_greed", "vix", "put_call_ratio"]
    ) -> Dict[str, Any]:
        """Get market sentiment indicators.
        
        Args:
            sources: List of sentiment sources to fetch
            
        Returns:
            Dictionary with sentiment indicators
        """
        
        sentiment_data = {}
        
        # CNN Fear & Greed Index
        if "fear_greed" in sources:
            try:
                url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
                response = self.session.get(url, timeout=10)
                data = response.json()
                
                latest_data = data['fear_and_greed'][-1]
                sentiment_data['fear_greed'] = {
                    'value': latest_data['value'],
                    'rating': latest_data['rating'],
                    'timestamp': latest_data['timestamp'],
                    'change': latest_data['value'] - data['fear_and_greed'][-2]['value'] if len(data['fear_and_greed']) > 1 else 0
                }
            except Exception as e:
                print(f"Warning: Could not fetch Fear & Greed index: {e}")
        
        # VIX Index (Volatility)
        if "vix" in sources:
            try:
                if YFINANCE_AVAILABLE:
                    vix_ticker = yf.Ticker("^VIX")
                    vix_data = vix_ticker.history(period="5d")
                    
                    if not vix_data.empty:
                        latest_vix = vix_data['Close'].iloc[-1]
                        sentiment_data['vix'] = {
                            'value': latest_vix,
                            'change_1d': (latest_vix - vix_data['Close'].iloc[-2]) / vix_data['Close'].iloc[-2],
                            'change_5d': (latest_vix - vix_data['Close'].iloc[0]) / vix_data['Close'].iloc[0],
                            'interpretation': 'High Fear' if latest_vix > 30 else 'Moderate' if latest_vix > 20 else 'Low Fear'
                        }
            except Exception as e:
                print(f"Warning: Could not fetch VIX data: {e}")
        
        # Put/Call Ratio
        if "put_call_ratio" in sources:
            try:
                # Using CBOE data (simplified approach)
                url = "https://www.cboe.com/us/options/market_statistics/daily/"
                response = self.session.get(url, timeout=10)
                
                # Parse HTML for put/call ratio (simplified)
                # In practice, you'd use proper HTML parsing
                sentiment_data['put_call_ratio'] = {
                    'value': 0.7,  # Placeholder
                    'interpretation': 'Bullish' if 0.7 < 0.8 else 'Bearish'
                }
            except Exception as e:
                print(f"Warning: Could not fetch Put/Call ratio: {e}")
        
        return sentiment_data
    
    def get_news_sentiment(
        self,
        keywords: List[str] = ["economy", "fed", "inflation", "recession"],
        max_articles: int = 50
    ) -> Dict[str, Any]:
        """Analyze news sentiment for economic topics.
        
        Args:
            keywords: List of keywords to search for
            max_articles: Maximum number of articles to analyze
            
        Returns:
            Dictionary with sentiment analysis
        """
        
        if not NEWSPAPER_AVAILABLE:
            print("Warning: Install newspaper3k for news sentiment: pip install newspaper3k")
            return {}
        
        articles_data = []
        sentiment_scores = []
        
        for keyword in keywords:
            try:
                # Search news (simplified - in practice, use news API)
                search_url = f"https://news.google.com/rss/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"
                response = self.session.get(search_url, timeout=10)
                
                # Parse RSS feed (simplified)
                import feedparser
                feed = feedparser.parse(response.content)
                
                for entry in feed.entries[:max_articles//len(keywords)]:
                    try:
                        article = Article(entry.link)
                        article.download()
                        article.parse()
                        
                        # Simple sentiment analysis
                        text = article.text.lower()
                        positive_words = ['growth', 'increase', 'strong', 'boost', 'recovery']
                        negative_words = ['decline', 'decrease', 'weak', 'fall', 'recession']
                        
                        positive_count = sum(1 for word in positive_words if word in text)
                        negative_count = sum(1 for word in negative_words if word in text)
                        
                        if positive_count + negative_count > 0:
                            sentiment = (positive_count - negative_count) / (positive_count + negative_count)
                        else:
                            sentiment = 0
                        
                        articles_data.append({
                            'title': article.title,
                            'source': article.source_url,
                            'published': entry.published,
                            'sentiment': sentiment,
                            'keyword': keyword
                        })
                        
                        sentiment_scores.append(sentiment)
                        
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"Warning: Could not fetch news for {keyword}: {e}")
        
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_interpretation = (
                'Very Positive' if avg_sentiment > 0.3 else
                'Positive' if avg_sentiment > 0.1 else
                'Neutral' if avg_sentiment > -0.1 else
                'Negative' if avg_sentiment > -0.3 else
                'Very Negative'
            )
        else:
            avg_sentiment = 0
            sentiment_interpretation = 'No Data'
        
        return {
            'average_sentiment': avg_sentiment,
            'interpretation': sentiment_interpretation,
            'article_count': len(articles_data),
            'articles': articles_data[:10]  # Return top 10 articles
        }
    
    def get_commodity_prices(
        self,
        commodities: List[str] = ["GC=F", "CL=F", "SI=F", "HG=F"]
    ) -> pd.DataFrame:
        """Get real-time commodity prices.
        
        Args:
            commodities: List of commodity symbols (Gold, Oil, Silver, Copper)
            
        Returns:
            DataFrame with commodity data
        """
        
        if not YFINANCE_AVAILABLE:
            raise ImportError("Install yfinance: pip install yfinance")
        
        return self.get_market_data_yahoo(commodities, period="5d", interval="1d")
    
    def get_currency_data(
        self,
        pairs: List[str] = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X"]
    ) -> pd.DataFrame:
        """Get real-time currency exchange rates.
        
        Args:
            pairs: List of currency pairs
            
        Returns:
            DataFrame with currency data
        """
        
        if not YFINANCE_AVAILABLE:
            raise ImportError("Install yfinance: pip install yfinance")
        
        return self.get_market_data_yahoo(pairs, period="5d", interval="1d")
    
    def get_yield_curve_data(self) -> pd.DataFrame:
        """Get current yield curve data.
        
        Returns:
            DataFrame with yield curve data
        """
        
        # Treasury yield symbols
        yields = {
            '2Y': '^FVX',
            '5Y': '^FVX',  # Using 5Y Treasury note futures
            '10Y': '^TNX',
            '30Y': '^TYX'
        }
        
        try:
            yield_data = self.get_market_data_yahoo(list(yields.values()), period="1d")
            
            # Extract current yields
            current_yields = {}
            for maturity, symbol in yields.items():
                if symbol in yield_data.columns.get_level_values(0):
                    current_yields[maturity] = yield_data[symbol]['Close'].iloc[-1]
            
            return pd.DataFrame(list(current_yields.items()), columns=['Maturity', 'Yield'])
            
        except Exception as e:
            print(f"Warning: Could not fetch yield curve data: {e}")
            return pd.DataFrame()
    
    def get_real_time_indicators(
        self,
        include_markets: bool = True,
        include_sentiment: bool = True,
        include_news: bool = True,
        include_commodities: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive real-time economic indicators.
        
        Args:
            include_markets: Include market data
            include_sentiment: Include sentiment indicators
            include_news: Include news sentiment
            include_commodities: Include commodity prices
            
        Returns:
            Dictionary with all real-time indicators
        """
        
        indicators = {
            'timestamp': datetime.now().isoformat(),
            'data_sources': []
        }
        
        # Market data
        if include_markets:
            try:
                market_symbols = ['^GSPC', '^DJI', '^IXIC', '^VIX']  # S&P 500, Dow, Nasdaq, VIX
                indicators['market_data'] = self.get_market_data_yahoo(market_symbols, period="1d")
                indicators['data_sources'].append('Yahoo Finance')
            except Exception as e:
                print(f"Warning: Market data fetch failed: {e}")
        
        # Sentiment indicators
        if include_sentiment:
            try:
                indicators['sentiment'] = self.get_market_sentiment()
                indicators['data_sources'].append('Sentiment APIs')
            except Exception as e:
                print(f"Warning: Sentiment data fetch failed: {e}")
        
        # News sentiment
        if include_news:
            try:
                indicators['news_sentiment'] = self.get_news_sentiment()
                indicators['data_sources'].append('News Sources')
            except Exception as e:
                print(f"Warning: News sentiment fetch failed: {e}")
        
        # Commodity prices
        if include_commodities:
            try:
                indicators['commodities'] = self.get_commodity_prices()
                indicators['data_sources'].append('Commodity Markets')
            except Exception as e:
                print(f"Warning: Commodity data fetch failed: {e}")
        
        # Economic calendar
        try:
            indicators['economic_calendar'] = self.get_economic_calendar()
            indicators['data_sources'].append('Economic Calendar')
        except Exception as e:
            print(f"Warning: Economic calendar fetch failed: {e}")
        
        # Yield curve
        try:
            indicators['yield_curve'] = self.get_yield_curve_data()
            indicators['data_sources'].append('Treasury Yields')
        except Exception as e:
            print(f"Warning: Yield curve fetch failed: {e}")
        
        return indicators
    
    def create_real_time_dataset(
        self,
        historical_data: pd.DataFrame,
        update_frequency: str = "daily"
    ) -> pd.DataFrame:
        """Create dataset combining historical and real-time data.
        
        Args:
            historical_data: Historical economic data
            update_frequency: Frequency of updates
            
        Returns:
            Combined dataset with real-time indicators
        """
        
        # Get real-time indicators
        realtime_indicators = self.get_real_time_indicators()
        
        # Create feature set from real-time data
        realtime_features = {}
        
        # Market features
        if 'market_data' in realtime_indicators:
            market_data = realtime_indicators['market_data']
            for symbol in ['^GSPC', '^DJI', '^IXIC']:
                if symbol in market_data.columns.get_level_values(0):
                    if len(market_data[symbol]) > 1:
                        returns = market_data[symbol]['Close'].pct_change().iloc[-1]
                        volatility = market_data[symbol]['Close'].pct_change().rolling(5).std().iloc[-1]
                        
                        realtime_features[f'{symbol}_return'] = returns
                        realtime_features[f'{symbol}_volatility'] = volatility
        
        # Sentiment features
        if 'sentiment' in realtime_indicators:
            sentiment = realtime_indicators['sentiment']
            if 'fear_greed' in sentiment:
                realtime_features['fear_greed_index'] = sentiment['fear_greed']['value']
            if 'vix' in sentiment:
                realtime_features['vix_level'] = sentiment['vix']['value']
        
        # Commodity features
        if 'commodities' in realtime_indicators:
            commodity_data = realtime_indicators['commodities']
            for symbol in ['GC=F', 'CL=F']:  # Gold, Oil
                if symbol in commodity_data.columns.get_level_values(0):
                    if len(commodity_data[symbol]) > 1:
                        price_change = commodity_data[symbol]['Close'].pct_change().iloc[-1]
                        realtime_features[f'{symbol}_change'] = price_change
        
        # Create combined dataset
        current_date = datetime.now().date()
        
        # Add real-time features to historical data
        combined_data = historical_data.copy()
        
        # Create row for current date
        current_row = pd.Series(realtime_features, name=current_date)
        combined_data = pd.concat([combined_data, current_row.to_frame().T])
        
        return combined_data


def get_alternative_data_indicators() -> Dict[str, Any]:
    """Get alternative data indicators for economic analysis.
    
    Returns:
        Dictionary with alternative data indicators
    """
    
    indicators = {}
    
    # Google Trends (simplified - would need API key in practice)
    try:
        # This would normally use pytrends library
        indicators['google_trends'] = {
            'economic_uncertainty': 75,  # Placeholder
            'job_searches': 60,          # Placeholder
            'inflation_concerns': 82     # Placeholder
        }
    except Exception:
        pass
    
    # Mobility data (simplified)
    try:
        indicators['mobility'] = {
            'retail_recreation': 95,     # Percentage of baseline
            'grocery_pharmacy': 102,
            'workplaces': 88,
            'transit_stations': 76
        }
    except Exception:
        pass
    
    # Supply chain indicators
    try:
        indicators['supply_chain'] = {
            'shipping_costs': 120,       # Container shipping index
            'delivery_times': 2.3,       # Average days
            'inventory_levels': 85       # Percentage of normal
        }
    except Exception:
        pass
    
    return indicators
