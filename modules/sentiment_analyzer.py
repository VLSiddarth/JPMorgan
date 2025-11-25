"""
Sentiment Analyzer Module
Analyzes news sentiment using FinBERT
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        self.newsapi_key = os.getenv('NEWSAPI_KEY', 'demo')
        self.sentiment_pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize FinBERT model (lazy loading)"""
        try:
            from transformers import pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                max_length=512,
                truncation=True
            )
            print("✅ FinBERT model loaded")
        except Exception as e:
            print(f"⚠️ Could not load FinBERT model: {e}")
            print("📊 Using fallback sentiment analysis")
            self.sentiment_pipeline = None
    
    def get_europe_news(self, days=7):
        """Fetch recent Europe-related news"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': '(Europe economy OR German fiscal OR ECB OR European Central Bank)',
                'apiKey': self.newsapi_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                return articles
            else:
                print(f"⚠️ NewsAPI error: {response.status_code}")
                return self._get_fallback_news()
                
        except Exception as e:
            print(f"⚠️ Error fetching news: {e}")
            return self._get_fallback_news()
    
    def _get_fallback_news(self):
        """Generate fallback news data"""
        return [
            {
                'title': 'European markets show resilience amid global uncertainty',
                'description': 'European equities demonstrate strength as investors focus on fiscal stimulus.',
                'source': {'name': 'Financial Times'},
                'publishedAt': datetime.now().isoformat(),
                'url': 'https://example.com/news1'
            },
            {
                'title': 'ECB signals continued support for European economy',
                'description': 'European Central Bank maintains accommodative monetary policy stance.',
                'source': {'name': 'Reuters'},
                'publishedAt': (datetime.now() - timedelta(hours=6)).isoformat(),
                'url': 'https://example.com/news2'
            },
            {
                'title': 'German fiscal expansion expected to boost growth',
                'description': 'Analysts expect German infrastructure spending to support European growth.',
                'source': {'name': 'Bloomberg'},
                'publishedAt': (datetime.now() - timedelta(hours=12)).isoformat(),
                'url': 'https://example.com/news3'
            }
        ]
    
    def analyze_sentiment(self, articles):
        """Analyze sentiment of news articles"""
        sentiments = []
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            
            if self.sentiment_pipeline and text.strip():
                try:
                    result = self.sentiment_pipeline(text[:512])[0]
                    sentiment = result['label'].lower()
                    score = result['score']
                except Exception as e:
                    print(f"⚠️ Sentiment analysis error: {e}")
                    sentiment = 'neutral'
                    score = 0.5
            else:
                # Fallback: simple keyword-based sentiment
                sentiment, score = self._simple_sentiment(text)
            
            sentiments.append({
                'title': article.get('title', 'No title'),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'date': article.get('publishedAt', datetime.now().isoformat())[:10],
                'sentiment': sentiment,
                'score': score,
                'url': article.get('url', '')
            })
        
        return pd.DataFrame(sentiments)
    
    def _simple_sentiment(self, text):
        """Simple keyword-based sentiment (fallback)"""
        text_lower = text.lower()
        
        positive_words = ['growth', 'strong', 'resilient', 'positive', 'bullish', 
                          'optimistic', 'recovery', 'expansion', 'boost']
        negative_words = ['crisis', 'weak', 'negative', 'bearish', 'pessimistic',
                          'recession', 'decline', 'risk', 'concern']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'positive', 0.7
        elif neg_count > pos_count:
            return 'negative', 0.7
        else:
            return 'neutral', 0.5
    
    def calculate_sentiment_score(self, df):
        """Aggregate sentiment into single score (-100 to +100)"""
        if df.empty:
            return 0
        
        # Map sentiment to numeric values
        score_map = {'positive': 1, 'negative': -1, 'neutral': 0}
        df['numeric_sentiment'] = df['sentiment'].map(score_map) * df['score']
        
        # Weight recent news more heavily
        df['date'] = pd.to_datetime(df['date'])
        df['days_ago'] = (datetime.now() - df['date']).dt.days
        df['weight'] = 1 / (1 + df['days_ago'] / 7)  # Decay over weeks
        
        weighted_score = (df['numeric_sentiment'] * df['weight']).sum() / df['weight'].sum()
        
        return weighted_score * 100  # Scale to -100 to +100
    
    def get_sentiment_summary(self):
        """Get complete sentiment analysis"""
        print("📰 Fetching and analyzing news...")
        
        articles = self.get_europe_news()
        sentiment_df = self.analyze_sentiment(articles)
        sentiment_score = self.calculate_sentiment_score(sentiment_df)
        
        # Count by sentiment
        sentiment_counts = sentiment_df['sentiment'].value_counts().to_dict()
        
        return {
            'score': sentiment_score,
            'articles': sentiment_df,
            'counts': sentiment_counts,
            'total_articles': len(sentiment_df)
        }


if __name__ == "__main__":
    # Test sentiment analyzer
    analyzer = SentimentAnalyzer()
    summary = analyzer.get_sentiment_summary()
    
    print(f"\n✅ Sentiment Analysis Complete")
    print(f"Overall Score: {summary['score']:.1f}/100")
    print(f"Total Articles: {summary['total_articles']}")
    print(f"Breakdown: {summary['counts']}")