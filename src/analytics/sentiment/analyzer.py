"""
Enhanced Sentiment Analysis Engine
Analyzes financial news and generates sentiment scores for market analysis
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    text: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # negative, neutral, positive
    confidence: float  # 0 to 1
    keywords: List[str]
    entities: List[str]
    timestamp: datetime
    source: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'text': self.text[:200] + '...' if len(self.text) > 200 else self.text,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'confidence': self.confidence,
            'keywords': self.keywords,
            'entities': self.entities,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source
        }


class SentimentAnalyzer:
    """
    Main sentiment analysis engine
    Combines multiple approaches: lexicon-based, ML-based, and financial-specific
    """
    
    # Financial sentiment lexicons
    POSITIVE_WORDS = {
        'bullish', 'growth', 'profit', 'surge', 'rally', 'gain', 'rise', 'up',
        'strong', 'boost', 'recovery', 'positive', 'outperform', 'success',
        'upgrade', 'beat', 'exceed', 'impressive', 'robust', 'momentum',
        'breakthrough', 'stimulus', 'optimistic', 'favorable', 'expansion'
    }
    
    NEGATIVE_WORDS = {
        'bearish', 'loss', 'decline', 'crash', 'fall', 'drop', 'down', 'weak',
        'recession', 'crisis', 'risk', 'concern', 'negative', 'underperform',
        'downgrade', 'miss', 'disappointing', 'slowdown', 'contraction',
        'uncertainty', 'volatile', 'warning', 'threat', 'tariff', 'deficit'
    }
    
    # Financial entities for European context
    EUROPEAN_ENTITIES = {
        'ECB', 'European Central Bank', 'Euro Stoxx', 'STOXX 50', 'DAX',
        'Germany', 'France', 'Italy', 'Spain', 'Eurozone', 'EU',
        'Christine Lagarde', 'Bundesbank', 'Deutsche Bank', 'BNP Paribas',
        'ASML', 'LVMH', 'SAP', 'Siemens', 'TotalEnergies'
    }
    
    # Topic keywords
    STIMULUS_KEYWORDS = {
        'stimulus', 'fiscal', 'spending', 'infrastructure', 'investment',
        'government support', 'recovery fund', 'budget'
    }
    
    TARIFF_KEYWORDS = {
        'tariff', 'trade war', 'protectionism', 'import tax', 'export',
        'trade policy', 'trade tension'
    }
    
    MONETARY_POLICY_KEYWORDS = {
        'interest rate', 'monetary policy', 'ECB rate', 'central bank',
        'rate cut', 'rate hike', 'quantitative easing', 'inflation target'
    }
    
    def __init__(self, use_finbert: bool = False):
        """
        Initialize sentiment analyzer
        
        Args:
            use_finbert: Whether to use FinBERT model (requires transformers)
        """
        self.use_finbert = use_finbert
        self.finbert_model = None
        self.finbert_tokenizer = None
        
        if use_finbert:
            self._load_finbert()
    
    def _load_finbert(self):
        """Load FinBERT model for advanced sentiment analysis"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            model_name = "ProsusAI/finbert"
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.finbert_model.eval()
            
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}. Using lexicon-based analysis.")
            self.use_finbert = False
    
    def analyze(self, text: str, source: Optional[str] = None) -> SentimentResult:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            source: Source of the text
            
        Returns:
            SentimentResult object
        """
        if not text or len(text.strip()) == 0:
            return self._empty_result()
        
        # Use FinBERT if available
        if self.use_finbert and self.finbert_model:
            return self._analyze_with_finbert(text, source)
        
        # Fallback to lexicon-based analysis
        return self._analyze_lexicon(text, source)
    
    def _analyze_with_finbert(self, text: str, source: Optional[str] = None) -> SentimentResult:
        """Analyze using FinBERT model"""
        try:
            import torch
            
            # Tokenize
            inputs = self.finbert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [negative, neutral, positive]
            scores = predictions[0].numpy()
            
            # Convert to -1 to 1 scale
            sentiment_score = float(scores[2] - scores[0])  # positive - negative
            confidence = float(max(scores))
            
            # Determine label
            if sentiment_score > 0.3:
                label = 'positive'
            elif sentiment_score < -0.3:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Extract keywords and entities
            keywords = self._extract_keywords(text)
            entities = self._extract_entities(text)
            
            return SentimentResult(
                text=text,
                sentiment_score=sentiment_score,
                sentiment_label=label,
                confidence=confidence,
                keywords=keywords,
                entities=entities,
                timestamp=datetime.now(),
                source=source
            )
            
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return self._analyze_lexicon(text, source)
    
    def _analyze_lexicon(self, text: str, source: Optional[str] = None) -> SentimentResult:
        """Lexicon-based sentiment analysis"""
        # Preprocess text
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Count sentiment words
        positive_count = sum(1 for word in words if word in self.POSITIVE_WORDS)
        negative_count = sum(1 for word in words if word in self.NEGATIVE_WORDS)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            confidence = 0.3
        else:
            # Calculate score
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            confidence = min(total_sentiment_words / 10, 1.0)
        
        # Adjust for intensity words
        intensifiers = ['very', 'extremely', 'highly', 'significantly']
        if any(word in text_lower for word in intensifiers):
            sentiment_score *= 1.2
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # Determine label
        if sentiment_score > 0.2:
            label = 'positive'
        elif sentiment_score < -0.2:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Extract keywords and entities
        keywords = self._extract_keywords(text)
        entities = self._extract_entities(text)
        
        return SentimentResult(
            text=text,
            sentiment_score=sentiment_score,
            sentiment_label=label,
            confidence=confidence,
            keywords=keywords,
            entities=entities,
            timestamp=datetime.now(),
            source=source
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        keywords = []
        text_lower = text.lower()
        
        # Check for topic keywords
        if any(kw in text_lower for kw in self.STIMULUS_KEYWORDS):
            keywords.append('stimulus')
        if any(kw in text_lower for kw in self.TARIFF_KEYWORDS):
            keywords.append('tariff')
        if any(kw in text_lower for kw in self.MONETARY_POLICY_KEYWORDS):
            keywords.append('monetary_policy')
        
        # Extract important financial terms
        words = re.findall(r'\b\w+\b', text_lower)
        important_words = set(words) & (self.POSITIVE_WORDS | self.NEGATIVE_WORDS)
        keywords.extend(list(important_words)[:5])
        
        return keywords
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (companies, indices, people)"""
        entities = []
        
        for entity in self.EUROPEAN_ENTITIES:
            if entity.lower() in text.lower():
                entities.append(entity)
        
        return entities[:5]
    
    def analyze_batch(self, texts: List[str], sources: Optional[List[str]] = None) -> List[SentimentResult]:
        """
        Analyze multiple texts
        
        Args:
            texts: List of texts
            sources: Optional list of sources
            
        Returns:
            List of SentimentResult objects
        """
        if sources is None:
            sources = [None] * len(texts)
        
        results = []
        for text, source in zip(texts, sources):
            result = self.analyze(text, source)
            results.append(result)
        
        return results
    
    def aggregate_sentiment(self, results: List[SentimentResult],
                          time_window: Optional[timedelta] = None) -> Dict:
        """
        Aggregate sentiment from multiple results
        
        Args:
            results: List of sentiment results
            time_window: Optional time window to filter results
            
        Returns:
            Aggregated sentiment metrics
        """
        if not results:
            return {
                'overall_score': 0.0,
                'overall_label': 'neutral',
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'confidence': 0.0,
                'sample_size': 0
            }
        
        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now() - time_window
            results = [r for r in results if r.timestamp > cutoff]
        
        if not results:
            return self.aggregate_sentiment(results, None)
        
        # Calculate metrics
        scores = [r.sentiment_score for r in results]
        labels = [r.sentiment_label for r in results]
        confidences = [r.confidence for r in results]
        
        overall_score = np.mean(scores)
        
        if overall_score > 0.2:
            overall_label = 'positive'
        elif overall_score < -0.2:
            overall_label = 'negative'
        else:
            overall_label = 'neutral'
        
        label_counts = Counter(labels)
        total = len(labels)
        
        return {
            'overall_score': float(overall_score),
            'overall_label': overall_label,
            'positive_ratio': label_counts['positive'] / total,
            'negative_ratio': label_counts['negative'] / total,
            'neutral_ratio': label_counts['neutral'] / total,
            'confidence': float(np.mean(confidences)),
            'sample_size': total,
            'score_std': float(np.std(scores)),
            'top_keywords': self._aggregate_keywords(results),
            'top_entities': self._aggregate_entities(results)
        }
    
    def _aggregate_keywords(self, results: List[SentimentResult]) -> List[Tuple[str, int]]:
        """Aggregate and count keywords"""
        all_keywords = []
        for result in results:
            all_keywords.extend(result.keywords)
        
        keyword_counts = Counter(all_keywords)
        return keyword_counts.most_common(10)
    
    def _aggregate_entities(self, results: List[SentimentResult]) -> List[Tuple[str, int]]:
        """Aggregate and count entities"""
        all_entities = []
        for result in results:
            all_entities.extend(result.entities)
        
        entity_counts = Counter(all_entities)
        return entity_counts.most_common(10)
    
    def get_sentiment_trend(self, results: List[SentimentResult],
                          window_hours: int = 24) -> pd.DataFrame:
        """
        Calculate sentiment trend over time
        
        Args:
            results: List of sentiment results
            window_hours: Hours per window
            
        Returns:
            DataFrame with sentiment trend
        """
        if not results:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                'timestamp': r.timestamp,
                'score': r.sentiment_score,
                'confidence': r.confidence
            }
            for r in results
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Resample to windows
        trend = df.resample(f'{window_hours}H').agg({
            'score': 'mean',
            'confidence': 'mean'
        })
        
        # Add weighted score
        trend['weighted_score'] = trend['score'] * trend['confidence']
        
        return trend
    
    def _empty_result(self) -> SentimentResult:
        """Return empty sentiment result"""
        return SentimentResult(
            text="",
            sentiment_score=0.0,
            sentiment_label='neutral',
            confidence=0.0,
            keywords=[],
            entities=[],
            timestamp=datetime.now()
        )
    
    def classify_news_impact(self, text: str) -> Dict:
        """
        Classify the potential market impact of news
        
        Args:
            text: News text
            
        Returns:
            Impact classification
        """
        text_lower = text.lower()
        
        # Analyze sentiment
        sentiment = self.analyze(text)
        
        # Check for high-impact keywords
        high_impact_keywords = {
            'ecb', 'interest rate', 'policy', 'stimulus', 'tariff',
            'crisis', 'recession', 'gdp', 'inflation'
        }
        
        impact_score = sum(1 for kw in high_impact_keywords if kw in text_lower)
        
        # Classify impact
        if impact_score >= 3 and abs(sentiment.sentiment_score) > 0.5:
            impact = 'high'
        elif impact_score >= 2 or abs(sentiment.sentiment_score) > 0.3:
            impact = 'medium'
        else:
            impact = 'low'
        
        return {
            'impact': impact,
            'impact_score': impact_score,
            'sentiment': sentiment.to_dict(),
            'relevant_topics': sentiment.keywords
        }