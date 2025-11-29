"""
Sentiment Analysis Module
Complete sentiment analysis toolkit for financial news and market data

This module provides:
- SentimentAnalyzer: Main sentiment engine (lexicon + optional FinBERT)
- FinBERTAnalyzer: Advanced deep learning sentiment analysis
- NewsClassifier: Topic detection and relevance scoring
- NewsFilter: Filter news by criteria

Usage Examples:
--------------

Example 1: Basic Sentiment Analysis
>>> from sentiment import SentimentAnalyzer
>>> analyzer = SentimentAnalyzer()
>>> result = analyzer.analyze("German fiscal stimulus boosts European stocks")
>>> print(f"Sentiment: {result.sentiment_label}, Score: {result.sentiment_score:.2f}")

Example 2: Using FinBERT (Advanced)
>>> from sentiment import FinBERTAnalyzer
>>> finbert = FinBERTAnalyzer()
>>> output = finbert.predict("ECB raises interest rates to combat inflation")
>>> print(f"FinBERT: {output.label} (confidence: {output.confidence:.2f})")

Example 3: News Classification
>>> from sentiment import NewsClassifier
>>> classifier = NewsClassifier()
>>> result = classifier.classify(
...     title="Germany announces €500B infrastructure plan",
...     content="The German government has approved..."
... )
>>> print(f"Topics: {result.topics}")
>>> print(f"Relevance: {result.relevance_score:.2f}")
>>> print(f"Market Impact: {result.market_impact}")

Example 4: Batch Processing
>>> analyzer = SentimentAnalyzer()
>>> texts = [
...     "Euro Stoxx 50 rallies on stimulus news",
...     "Trade tensions weigh on European exporters",
...     "ECB maintains accommodative stance"
... ]
>>> results = analyzer.analyze_batch(texts)
>>> aggregate = analyzer.aggregate_sentiment(results)
>>> print(f"Overall sentiment: {aggregate['overall_label']}")

Example 5: Complete News Analysis Pipeline
>>> from sentiment import SentimentAnalyzer, NewsClassifier
>>> 
>>> # Get news articles (from your news connector)
>>> articles = [
...     {
...         'title': 'German stimulus to boost European growth',
...         'content': 'Germany approved €500B infrastructure...'
...     }
... ]
>>> 
>>> # Classify and filter
>>> classifier = NewsClassifier()
>>> relevant = classifier.filter_relevant_news(articles, min_relevance=0.5)
>>> 
>>> # Analyze sentiment
>>> analyzer = SentimentAnalyzer()
>>> for article in relevant:
...     sentiment = analyzer.analyze(article.title)
...     print(f"{article.title[:50]}... -> {sentiment.sentiment_label}")
"""

from .analyzer import SentimentAnalyzer, SentimentResult
from .finbert import FinBERTAnalyzer, FinBERTOutput, FinBERTSentimentTracker
from .news_classifier import NewsClassifier, NewsClassification, NewsFilter

__all__ = [
    # Main analyzer
    'SentimentAnalyzer',
    'SentimentResult',
    
    # FinBERT
    'FinBERTAnalyzer',
    'FinBERTOutput',
    'FinBERTSentimentTracker',
    
    # News classification
    'NewsClassifier',
    'NewsClassification',
    'NewsFilter'
]

__version__ = '1.0.0'


# =============================================================================
# Integration Example: European Stocks News Analysis
# =============================================================================

def analyze_european_news_sentiment(articles: list, use_finbert: bool = False):
    """
    Complete pipeline for analyzing European market news
    
    Args:
        articles: List of dicts with 'title' and 'content'
        use_finbert: Whether to use FinBERT (slower but more accurate)
        
    Returns:
        Dict with comprehensive analysis
    
    Example:
    >>> articles = [
    ...     {'title': 'ECB cuts rates', 'content': '...'},
    ...     {'title': 'German stimulus approved', 'content': '...'}
    ... ]
    >>> results = analyze_european_news_sentiment(articles)
    >>> print(results['overall_sentiment'])
    """
    # Initialize components
    classifier = NewsClassifier()
    
    if use_finbert:
        from .finbert import FinBERTAnalyzer
        sentiment_analyzer = FinBERTAnalyzer()
        
        # Analyze with FinBERT
        sentiment_results = sentiment_analyzer.analyze_news_batch(articles)
        
        # Get market sentiment
        titles = [a['title'] for a in articles]
        market_sentiment = sentiment_analyzer.get_market_sentiment(titles)
        
    else:
        sentiment_analyzer = SentimentAnalyzer()
        
        # Analyze with lexicon
        sentiment_results = []
        for article in articles:
            text = article['title'] + ' ' + article.get('content', '')[:500]
            result = sentiment_analyzer.analyze(text)
            sentiment_results.append(result.to_dict())
        
        # Aggregate
        all_results = [
            sentiment_analyzer.analyze(article['title'])
            for article in articles
        ]
        market_sentiment = sentiment_analyzer.aggregate_sentiment(all_results)
    
    # Classify news
    classifications = classifier.classify_batch(articles)
    topic_summary = classifier.get_topic_summary(classifications)
    
    # Filter high-impact news
    high_impact = [c for c in classifications if c.market_impact == 'high']
    
    # German stimulus specific (JPMorgan key theme)
    stimulus_news = [
        c for c in classifications
        if 'german_stimulus' in c.topics or 'fiscal_policy' in c.topics
    ]
    
    return {
        'overall_sentiment': market_sentiment,
        'topic_summary': topic_summary,
        'high_impact_count': len(high_impact),
        'high_impact_news': [c.to_dict() for c in high_impact[:5]],
        'stimulus_news_count': len(stimulus_news),
        'stimulus_news': [c.to_dict() for c in stimulus_news[:5]],
        'all_sentiment_results': sentiment_results[:10],  # Top 10
        'analysis_timestamp': __import__('datetime').datetime.now().isoformat()
    }


def get_sentiment_signal(articles: list, threshold: float = 0.3):
    """
    Generate trading signal based on news sentiment
    
    Args:
        articles: List of news articles
        threshold: Sentiment threshold for signals
        
    Returns:
        Dict with signal and reasoning
        
    Example:
    >>> signal = get_sentiment_signal(recent_news)
    >>> print(f"Signal: {signal['action']} - {signal['reason']}")
    """
    analysis = analyze_european_news_sentiment(articles)
    
    overall = analysis['overall_sentiment']
    score = overall['overall_score']
    confidence = overall['confidence']
    
    # Generate signal
    if score > threshold and confidence > 0.6:
        action = 'BUY'
        strength = 'STRONG' if score > 0.5 else 'MODERATE'
        reason = f"Positive sentiment ({score:.2f}) with high confidence"
    elif score < -threshold and confidence > 0.6:
        action = 'SELL'
        strength = 'STRONG' if score < -0.5 else 'MODERATE'
        reason = f"Negative sentiment ({score:.2f}) with high confidence"
    else:
        action = 'HOLD'
        strength = 'NEUTRAL'
        reason = f"Mixed or uncertain sentiment (score: {score:.2f}, confidence: {confidence:.2f})"
    
    return {
        'action': action,
        'strength': strength,
        'reason': reason,
        'sentiment_score': score,
        'confidence': confidence,
        'high_impact_news_count': analysis['high_impact_count'],
        'key_topics': [topic for topic, count in analysis['topic_summary']['top_topics'][:3]]
    }


# =============================================================================
# Quick Start Functions
# =============================================================================

def quick_sentiment(text: str) -> dict:
    """
    Quick sentiment analysis for a single text
    
    Args:
        text: Text to analyze
        
    Returns:
        Simple sentiment dict
    """
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze(text)
    
    return {
        'label': result.sentiment_label,
        'score': result.sentiment_score,
        'confidence': result.confidence
    }


def quick_news_analysis(title: str, content: str = None) -> dict:
    """
    Quick news classification
    
    Args:
        title: News title
        content: News content (optional)
        
    Returns:
        Classification dict
    """
    classifier = NewsClassifier()
    result = classifier.classify(title, content)
    
    return {
        'topics': result.topics,
        'relevance': result.relevance_score,
        'impact': result.market_impact,
        'sentiment_hint': result.sentiment_hint,
        'regions': result.regions
    }


# =============================================================================
# Testing Utilities
# =============================================================================

def test_sentiment_module():
    """Test all components of sentiment module"""
    print("Testing Sentiment Analysis Module...")
    print("=" * 60)
    
    # Test 1: Basic sentiment
    print("\n1. Testing Basic Sentiment Analysis")
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "European stocks rally on German stimulus hopes",
        "Trade tensions weigh on export-heavy sectors",
        "ECB maintains interest rates at current levels"
    ]
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"  '{text[:50]}...'")
        print(f"  -> {result.sentiment_label.upper()} ({result.sentiment_score:+.2f})")
    
    # Test 2: News classification
    print("\n2. Testing News Classification")
    classifier = NewsClassifier()
    
    test_article = {
        'title': 'Germany announces €500B infrastructure stimulus package',
        'content': 'The German government has approved a landmark fiscal reform...'
    }
    
    classification = classifier.classify(test_article['title'], test_article['content'])
    print(f"  Title: {test_article['title']}")
    print(f"  Topics: {', '.join(classification.topics)}")
    print(f"  Relevance: {classification.relevance_score:.2f}")
    print(f"  Impact: {classification.market_impact.upper()}")
    print(f"  Regions: {', '.join(classification.regions)}")
    
    # Test 3: Batch processing
    print("\n3. Testing Batch Analysis")
    results = analyzer.analyze_batch(test_texts)
    aggregate = analyzer.aggregate_sentiment(results)
    
    print(f"  Overall Sentiment: {aggregate['overall_label'].upper()}")
    print(f"  Overall Score: {aggregate['overall_score']:+.2f}")
    print(f"  Confidence: {aggregate['confidence']:.2f}")
    print(f"  Sample Size: {aggregate['sample_size']}")
    
    # Test 4: Signal generation
    print("\n4. Testing Signal Generation")
    signal = get_sentiment_signal([test_article])
    print(f"  Action: {signal['action']} ({signal['strength']})")
    print(f"  Reason: {signal['reason']}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_sentiment_module()