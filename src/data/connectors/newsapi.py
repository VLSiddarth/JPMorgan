"""
NewsAPI Connector - Financial News & Sentiment
Access European financial news with sentiment analysis

Features:
- Real-time European financial news
- Multi-source aggregation (FT, Bloomberg, Reuters, WSJ)
- Keyword filtering and relevance scoring
- Sentiment analysis integration-ready
- Caching and rate limiting
- Free tier: 100 requests/day (1000/day for students!)

API: https://newsapi.org
Get FREE key: https://newsapi.org/register

Author: JPMorgan Dashboard Team
"""

import requests
import pandas as pd
from typing import Optional, List, Dict, Union
from datetime import datetime, timedelta
from pathlib import Path
import time
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from config.settings import settings

logger = logging.getLogger(__name__)


class NewsCategory(Enum):
    """News categories"""
    BUSINESS = 'business'
    TECHNOLOGY = 'technology'
    GENERAL = 'general'


class NewsSortBy(Enum):
    """Sorting options"""
    RELEVANCY = 'relevancy'
    POPULARITY = 'popularity'
    PUBLISHED_AT = 'publishedAt'


@dataclass
class NewsArticle:
    """Structured news article"""
    title: str
    description: Optional[str]
    url: str
    source: str
    published_at: datetime
    content: Optional[str] = None
    author: Optional[str] = None
    image_url: Optional[str] = None
    
    # Analysis fields (populated by sentiment analyzer)
    sentiment_score: Optional[float] = None  # -100 to +100
    sentiment_label: Optional[str] = None    # 'positive', 'negative', 'neutral'
    relevance_score: Optional[float] = None  # 0 to 1
    keywords: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['published_at'] = self.published_at.isoformat()
        return data
    
    def get_id(self) -> str:
        """Generate unique ID for article"""
        unique_str = f"{self.source}_{self.title}_{self.published_at.isoformat()}"
        return hashlib.md5(unique_str.encode()).hexdigest()


class NewsAPIConnector:
    """
    NewsAPI Connector for European Financial News
    
    Features:
    - Multi-source news aggregation
    - Keyword and category filtering
    - Date range queries
    - Pagination support
    - Automatic caching
    - Rate limiting
    
    Example:
        >>> news = NewsAPIConnector()
        >>> articles = news.get_european_news(
        ...     keywords=['ECB', 'Europe economy'],
        ...     days=7
        ... )
        >>> for article in articles:
        ...     print(f"{article.title} - {article.source}")
    """
    
    BASE_URL = 'https://newsapi.org/v2'
    
    # Trusted financial sources
    FINANCIAL_SOURCES = [
        'financial-times',
        'bloomberg',
        'reuters',
        'the-wall-street-journal',
        'business-insider',
        'cnbc',
        'fortune',
        'the-economist'
    ]
    
    # European keywords
    DEFAULT_KEYWORDS = [
        'Europe economy',
        'European Central Bank',
        'ECB',
        'STOXX',
        'Euro area',
        'European markets',
        'German fiscal',
        'EU policy',
        'Eurozone'
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        timeout: int = 15
    ):
        """
        Initialize NewsAPI connector
        
        Args:
            api_key: NewsAPI key (from settings if not provided)
            cache_dir: Cache directory
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or settings.NEWSAPI_KEY
        self.cache_dir = cache_dir or settings.CACHE_DIR / 'news'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        
        # Request tracking for rate limiting
        self.request_times = []
        self.daily_requests = 0
        self.last_reset = datetime.now().date()
        
        if not self.api_key or self.api_key in ['demo', 'your_newsapi_key_here', '']:
            logger.warning(
                "⚠️  NewsAPI key not configured. "
                "Get FREE key: https://newsapi.org/register"
            )
            self.enabled = False
        else:
            self.enabled = True
            logger.info("✅ NewsAPI initialized")
    
    def _check_rate_limit(self):
        """Check and enforce rate limits"""
        
        # Reset daily counter
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_requests = 0
            self.last_reset = today
        
        # Check daily limit (100 for free tier, 1000 for developer)
        max_daily = 100  # Conservative limit
        if self.daily_requests >= max_daily:
            logger.error(f"Daily rate limit reached ({max_daily} requests)")
            raise Exception("NewsAPI daily rate limit exceeded")
        
        # Check per-second limit (avoid rapid requests)
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 1]
        
        if len(self.request_times) >= 2:  # Max 2 requests per second
            sleep_time = 1 - (now - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.request_times.append(time.time())
        self.daily_requests += 1
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key from request parameters"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{endpoint}_{param_str}".encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_cache(self, cache_key: str, max_age_hours: int = 6) -> Optional[List[Dict]]:
        """Load from cache if recent"""
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age > max_age_hours * 3600:
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"📂 Loaded {len(data.get('articles', []))} articles from cache")
            return data.get('articles', [])
            
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save to cache"""
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                'cached_at': datetime.now().isoformat(),
                'articles': data.get('articles', [])
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Cached {len(cache_data['articles'])} articles")
            
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def _make_request(
        self,
        endpoint: str,
        params: Dict,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Make API request with caching and rate limiting
        
        Args:
            endpoint: API endpoint ('everything' or 'top-headlines')
            params: Query parameters
            use_cache: Use cache if available
            
        Returns:
            API response dictionary
        """
        
        if not self.enabled:
            logger.warning("NewsAPI not enabled (no API key)")
            return self._get_dummy_news()
        
        # Check cache first
        cache_key = self._get_cache_key(endpoint, params)
        
        if use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return {'articles': cached, 'status': 'ok'}
        
        # Rate limiting
        try:
            self._check_rate_limit()
        except Exception as e:
            logger.error(f"Rate limit error: {e}")
            return None
        
        # Add API key to params
        params['apiKey'] = self.api_key
        
        # Make request
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            logger.debug(f"Fetching news: {endpoint}")
            response = requests.get(
                url,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') == 'ok':
                    # Cache successful response
                    if use_cache:
                        self._save_to_cache(cache_key, data)
                    
                    return data
                else:
                    logger.error(f"NewsAPI error: {data.get('message', 'Unknown')}")
                    return None
            
            elif response.status_code == 429:
                logger.error("NewsAPI rate limit exceeded")
                return None
            
            elif response.status_code == 401:
                logger.error("NewsAPI authentication failed (invalid API key)")
                return None
            
            else:
                logger.error(f"NewsAPI HTTP {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("NewsAPI request timeout")
            return None
        
        except Exception as e:
            logger.error(f"NewsAPI request failed: {e}")
            return None
    
    def _parse_article(self, raw: Dict) -> NewsArticle:
        """Parse raw article data to NewsArticle object"""
        
        try:
            return NewsArticle(
                title=raw.get('title', 'Untitled'),
                description=raw.get('description'),
                url=raw.get('url', ''),
                source=raw.get('source', {}).get('name', 'Unknown'),
                published_at=pd.to_datetime(raw.get('publishedAt')),
                content=raw.get('content'),
                author=raw.get('author'),
                image_url=raw.get('urlToImage')
            )
        except Exception as e:
            logger.warning(f"Failed to parse article: {e}")
            return None
    
    def get_european_news(
        self,
        keywords: Optional[List[str]] = None,
        days: int = 7,
        sources: Optional[List[str]] = None,
        max_results: int = 100,
        language: str = 'en',
        sort_by: NewsSortBy = NewsSortBy.PUBLISHED_AT,
        use_cache: bool = True
    ) -> List[NewsArticle]:
        """
        Get European financial news
        
        Args:
            keywords: List of keywords to search (OR logic)
            days: Number of days to look back
            sources: News sources (default: financial sources)
            max_results: Maximum articles to return
            language: Language code (default: 'en')
            sort_by: Sort order
            use_cache: Use cached results
            
        Returns:
            List of NewsArticle objects
            
        Example:
            >>> news = NewsAPIConnector()
            >>> articles = news.get_european_news(
            ...     keywords=['ECB', 'German fiscal'],
            ...     days=7,
            ...     max_results=50
            ... )
            >>> print(f"Found {len(articles)} articles")
        """
        
        if keywords is None:
            keywords = self.DEFAULT_KEYWORDS
        
        if sources is None:
            sources = self.FINANCIAL_SOURCES
        
        # Build query
        query = ' OR '.join(f'"{kw}"' for kw in keywords)
        
        # Calculate date range
        from_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        params = {
            'q': query,
            'sources': ','.join(sources),
            'from': from_date,
            'language': language,
            'sortBy': sort_by.value,
            'pageSize': min(max_results, 100)  # API limit is 100 per request
        }
        
        # Make request
        response = self._make_request('everything', params, use_cache)
        
        if not response or response.get('status') != 'ok':
            logger.warning("No news data returned")
            return []
        
        # Parse articles
        raw_articles = response.get('articles', [])
        articles = []
        
        for raw in raw_articles:
            article = self._parse_article(raw)
            if article:
                articles.append(article)
        
        logger.info(f"✅ Fetched {len(articles)} news articles")
        return articles
    
    def get_top_headlines(
        self,
        category: NewsCategory = NewsCategory.BUSINESS,
        country: str = 'de',  # Germany as proxy for European news
        max_results: int = 20
    ) -> List[NewsArticle]:
        """
        Get top headlines
        
        Args:
            category: News category
            country: Country code (de, fr, it, es, gb)
            max_results: Maximum articles
            
        Returns:
            List of NewsArticle objects
        """
        
        params = {
            'category': category.value,
            'country': country,
            'pageSize': min(max_results, 100)
        }
        
        response = self._make_request('top-headlines', params)
        
        if not response or response.get('status') != 'ok':
            return []
        
        raw_articles = response.get('articles', [])
        articles = [self._parse_article(raw) for raw in raw_articles]
        
        return [a for a in articles if a is not None]
    
    def get_sentiment_data(
        self,
        keywords: Optional[List[str]] = None,
        days: int = 7
    ) -> Dict:
        """
        Get aggregated sentiment data
        
        Args:
            keywords: Keywords to search
            days: Days to look back
            
        Returns:
            Dictionary with sentiment metrics
        """
        
        articles = self.get_european_news(keywords=keywords, days=days)
        
        if not articles:
            return {
                'total_articles': 0,
                'sentiment': 'neutral',
                'confidence': 0.0
            }
        
        # Note: Actual sentiment analysis happens in sentiment analyzer module
        # This is just a placeholder structure
        
        return {
            'total_articles': len(articles),
            'date_range': {
                'from': min(a.published_at for a in articles).isoformat(),
                'to': max(a.published_at for a in articles).isoformat()
            },
            'sources': list(set(a.source for a in articles)),
            'articles': [a.to_dict() for a in articles[:10]]  # Top 10 for preview
        }
    
    def _get_dummy_news(self) -> Dict:
        """Fallback dummy news when API unavailable"""
        
        logger.warning("Using dummy news data")
        
        dummy_articles = [
            {
                'title': 'European Markets Show Resilience Amid Economic Uncertainty',
                'description': 'European equity markets demonstrated strength in recent trading.',
                'source': {'name': 'Financial Times'},
                'publishedAt': datetime.now().isoformat(),
                'url': 'https://example.com/news1',
                'content': 'European markets continue to perform well...'
            },
            {
                'title': 'ECB Maintains Accommodative Policy Stance',
                'description': 'The European Central Bank kept rates steady at current levels.',
                'source': {'name': 'Bloomberg'},
                'publishedAt': (datetime.now() - timedelta(days=1)).isoformat(),
                'url': 'https://example.com/news2',
                'content': 'ECB policymakers agreed to maintain...'
            },
            {
                'title': 'German Fiscal Stimulus Boosts Growth Outlook',
                'description': 'Germanys infrastructure spending plan exceeds expectations.',
                'source': {'name': 'Reuters'},
                'publishedAt': (datetime.now() - timedelta(days=2)).isoformat(),
                'url': 'https://example.com/news3',
                'content': 'The German governments fiscal package...'
            }
        ]
        
        return {
            'status': 'ok',
            'articles': dummy_articles
        }
    
    def health_check(self) -> Dict[str, any]:
        """
        Check NewsAPI health
        
        Returns:
            Health status dictionary
        """
        
        status = {
            'newsapi': 'unavailable',
            'api_key_configured': bool(self.api_key and self.api_key not in ['demo', '']),
            'enabled': self.enabled,
            'cache_dir': str(self.cache_dir),
            'cached_queries': len(list(self.cache_dir.glob('*.json'))),
            'daily_requests': self.daily_requests,
            'test_query': False
        }
        
        if self.enabled:
            try:
                # Test with minimal query
                test = self._make_request(
                    'top-headlines',
                    {'country': 'de', 'pageSize': 1},
                    use_cache=False
                )
                
                if test and test.get('status') == 'ok':
                    status['newsapi'] = 'healthy'
                    status['test_query'] = True
                    articles = test.get('articles', [])
                    if articles:
                        status['latest_headline'] = articles[0].get('title', '')
                        
            except Exception as e:
                status['newsapi'] = f'error: {str(e)}'
        
        return status


if __name__ == "__main__":
    # Test NewsAPI connector
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("TESTING NEWSAPI CONNECTOR")
    print("="*70)
    
    news = NewsAPIConnector()
    
    # Test 1: Health check
    print("\n[Test 1] Health Check:")
    health = news.health_check()
    for key, value in health.items():
        print(f"  {key}: {value}")
    
    if news.enabled:
        # Test 2: Get European news
        print("\n[Test 2] European Financial News (last 7 days):")
        articles = news.get_european_news(
            keywords=['Europe economy', 'ECB'],
            days=7,
            max_results=10
        )
        
        print(f"  Found {len(articles)} articles")
        if articles:
            print("\n  Latest articles:")
            for i, article in enumerate(articles[:5], 1):
                print(f"    {i}. {article.title}")
                print(f"       Source: {article.source} | Date: {article.published_at.date()}")
        
        # Test 3: Get top headlines
        print("\n[Test 3] Top Business Headlines (Germany):")
        headlines = news.get_top_headlines(
            category=NewsCategory.BUSINESS,
            country='de',
            max_results=5
        )
        
        print(f"  Found {len(headlines)} headlines")
        for i, article in enumerate(headlines, 1):
            print(f"    {i}. {article.title}")
    
    else:
        print("\n⚠️  NewsAPI not enabled. Configure API key to test.")
    
    print("\n" + "="*70)
    print("✅ NEWSAPI CONNECTOR TEST COMPLETE")
    print("="*70)