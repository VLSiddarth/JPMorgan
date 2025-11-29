"""
News Classification and Topic Detection
Classifies news articles by topic, relevance, and market impact
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NewsClassification:
    """News article classification result"""
    title: str
    topics: List[str]
    relevance_score: float  # 0 to 1
    market_impact: str  # low, medium, high
    sentiment_hint: str  # positive, negative, neutral
    entities: List[str]
    regions: List[str]
    sectors: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'topics': self.topics,
            'relevance_score': self.relevance_score,
            'market_impact': self.market_impact,
            'sentiment_hint': self.sentiment_hint,
            'entities': self.entities,
            'regions': self.regions,
            'sectors': self.sectors,
            'timestamp': self.timestamp.isoformat()
        }


class NewsClassifier:
    """
    Classify news articles by topic, relevance, and potential market impact
    Focused on European markets and JPMorgan research themes
    """
    
    # Topic keyword mappings
    TOPIC_KEYWORDS = {
        'monetary_policy': {
            'ecb', 'european central bank', 'interest rate', 'monetary policy',
            'rate cut', 'rate hike', 'quantitative easing', 'lagarde',
            'inflation target', 'policy rate', 'deposit rate'
        },
        'fiscal_policy': {
            'fiscal', 'budget', 'stimulus', 'infrastructure', 'spending',
            'government', 'deficit', 'debt brake', 'fiscal expansion',
            'recovery fund', 'investment plan'
        },
        'german_stimulus': {
            'germany', 'german', 'bundesbank', 'scholz', 'berlin',
            '500 billion', 'infrastructure fund', 'debt brake reform',
            'constitutional reform'
        },
        'trade_tariffs': {
            'tariff', 'trade war', 'protectionism', 'import', 'export',
            'trade policy', 'customs', 'trade deal', 'wto',
            'trade tension', 'us tariff', 'china tariff'
        },
        'china_economy': {
            'china', 'chinese', 'beijing', 'pmi', 'chinese economy',
            'yuan', 'pboc', 'property crisis', 'stimulus china',
            'chinese growth', 'manufacturing china'
        },
        'defense_spending': {
            'defense', 'military', 'nato', 'arms', 'aerospace',
            'rheinmetall', 'bae systems', 'leonardo', 'thales',
            'defense budget', 'rearmament', 'security'
        },
        'banking_sector': {
            'bank', 'banking', 'financial', 'credit', 'loan',
            'deutsche bank', 'bnp paribas', 'unicredit', 'santander',
            'net interest margin', 'loan growth'
        },
        'corporate_earnings': {
            'earnings', 'profit', 'revenue', 'eps', 'guidance',
            'quarterly results', 'beat estimates', 'miss estimates',
            'earnings season', 'outlook'
        },
        'market_performance': {
            'euro stoxx', 'stoxx 50', 'dax', 'cac 40', 'ftse',
            'stock market', 'equity', 'index', 'rally', 'selloff',
            'market close', 'trading'
        }
    }
    
    # Entity keywords
    COMPANIES = {
        'ASML', 'LVMH', 'Novo Nordisk', 'SAP', 'Novartis', 'Roche',
        'Nestle', 'AstraZeneca', "L'Oreal", 'Siemens', 'TotalEnergies',
        'Rheinmetall', 'Airbus', 'Deutsche Bank', 'BNP Paribas'
    }
    
    REGIONS = {
        'Germany': {'germany', 'german', 'berlin', 'munich', 'frankfurt'},
        'France': {'france', 'french', 'paris', 'macron'},
        'Italy': {'italy', 'italian', 'rome', 'milan'},
        'Spain': {'spain', 'spanish', 'madrid', 'barcelona'},
        'UK': {'uk', 'britain', 'british', 'london', 'england'},
        'Eurozone': {'eurozone', 'euro area', 'eu', 'european union'},
        'US': {'us', 'usa', 'united states', 'america', 'american'},
        'China': {'china', 'chinese', 'beijing', 'shanghai'}
    }
    
    SECTORS = {
        'Technology': {'tech', 'technology', 'software', 'semiconductor', 'digital'},
        'Financials': {'bank', 'banking', 'financial', 'insurance'},
        'Healthcare': {'healthcare', 'pharma', 'pharmaceutical', 'biotech'},
        'Industrials': {'industrial', 'manufacturing', 'aerospace', 'defense'},
        'Consumer': {'consumer', 'retail', 'luxury', 'goods'},
        'Energy': {'energy', 'oil', 'gas', 'renewable'},
        'Utilities': {'utility', 'utilities', 'power', 'electricity'}
    }
    
    # High-impact indicators
    HIGH_IMPACT_INDICATORS = {
        'breaking', 'urgent', 'alert', 'crisis', 'crash', 'surge',
        'record', 'historic', 'unprecedented', 'major', 'significant',
        'announces', 'decision', 'policy change'
    }
    
    def __init__(self):
        """Initialize news classifier"""
        pass
    
    def classify(self, title: str, content: Optional[str] = None) -> NewsClassification:
        """
        Classify news article
        
        Args:
            title: Article title
            content: Article content (optional)
            
        Returns:
            NewsClassification object
        """
        text = title
        if content:
            text += " " + content[:500]  # Use first 500 chars of content
        
        text_lower = text.lower()
        
        # Detect topics
        topics = self._detect_topics(text_lower)
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance(text_lower, topics)
        
        # Assess market impact
        market_impact = self._assess_market_impact(text_lower, topics)
        
        # Get sentiment hint (not full sentiment analysis)
        sentiment_hint = self._get_sentiment_hint(text_lower)
        
        # Extract entities
        entities = self._extract_companies(text)
        
        # Detect regions
        regions = self._detect_regions(text_lower)
        
        # Detect sectors
        sectors = self._detect_sectors(text_lower)
        
        return NewsClassification(
            title=title,
            topics=topics,
            relevance_score=relevance_score,
            market_impact=market_impact,
            sentiment_hint=sentiment_hint,
            entities=entities,
            regions=regions,
            sectors=sectors,
            timestamp=datetime.now()
        )
    
    def _detect_topics(self, text: str) -> List[str]:
        """Detect topics in text"""
        detected_topics = []
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in text)
            
            # If enough matches, include topic
            if matches >= 1:  # At least 1 keyword match
                detected_topics.append(topic)
        
        return detected_topics
    
    def _calculate_relevance(self, text: str, topics: List[str]) -> float:
        """
        Calculate relevance score for European stocks analysis
        
        High relevance: Multiple key topics, European focus
        Low relevance: Generic news, non-European focus
        """
        score = 0.0
        
        # Topic relevance
        high_value_topics = {
            'german_stimulus', 'monetary_policy', 'fiscal_policy',
            'defense_spending', 'banking_sector'
        }
        
        for topic in topics:
            if topic in high_value_topics:
                score += 0.25
            else:
                score += 0.15
        
        # European focus bonus
        european_mentions = sum(
            1 for region in ['Germany', 'France', 'Italy', 'Spain', 'Eurozone']
            if any(kw in text for kw in self.REGIONS[region])
        )
        
        if european_mentions >= 2:
            score += 0.3
        elif european_mentions >= 1:
            score += 0.15
        
        # Key index mentions
        if any(idx in text for idx in ['euro stoxx', 'stoxx 50', 'dax']):
            score += 0.2
        
        # Company mentions (GRANOLAS, etc.)
        company_mentions = sum(1 for company in self.COMPANIES if company.lower() in text)
        score += min(company_mentions * 0.1, 0.3)
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _assess_market_impact(self, text: str, topics: List[str]) -> str:
        """
        Assess potential market impact
        
        Returns: 'low', 'medium', or 'high'
        """
        impact_score = 0
        
        # High-impact topics
        high_impact_topics = {
            'monetary_policy', 'trade_tariffs', 'german_stimulus'
        }
        
        impact_score += sum(2 for topic in topics if topic in high_impact_topics)
        impact_score += sum(1 for topic in topics if topic not in high_impact_topics)
        
        # High-impact indicators
        impact_score += sum(2 for indicator in self.HIGH_IMPACT_INDICATORS if indicator in text)
        
        # ECB or major central bank
        if 'ecb' in text or 'european central bank' in text:
            impact_score += 2
        
        # Major policy changes
        policy_words = ['announces', 'decision', 'policy', 'change', 'reform']
        if sum(1 for word in policy_words if word in text) >= 2:
            impact_score += 1
        
        # Classify
        if impact_score >= 6:
            return 'high'
        elif impact_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _get_sentiment_hint(self, text: str) -> str:
        """
        Get quick sentiment hint (not full analysis)
        
        Returns: 'positive', 'negative', or 'neutral'
        """
        positive_words = {
            'gain', 'rise', 'growth', 'boost', 'rally', 'surge',
            'up', 'bullish', 'strong', 'positive', 'optimistic',
            'recovery', 'expansion', 'improve'
        }
        
        negative_words = {
            'fall', 'drop', 'decline', 'crash', 'loss', 'down',
            'bearish', 'weak', 'negative', 'pessimistic', 'crisis',
            'recession', 'slowdown', 'concern', 'risk'
        }
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count + 1:
            return 'positive'
        elif neg_count > pos_count + 1:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_companies(self, text: str) -> List[str]:
        """Extract company mentions"""
        mentioned = []
        
        for company in self.COMPANIES:
            # Case-insensitive search
            if re.search(r'\b' + re.escape(company) + r'\b', text, re.IGNORECASE):
                mentioned.append(company)
        
        return mentioned
    
    def _detect_regions(self, text: str) -> List[str]:
        """Detect geographic regions mentioned"""
        detected = []
        
        for region, keywords in self.REGIONS.items():
            if any(kw in text for kw in keywords):
                detected.append(region)
        
        return detected
    
    def _detect_sectors(self, text: str) -> List[str]:
        """Detect sectors mentioned"""
        detected = []
        
        for sector, keywords in self.SECTORS.items():
            if any(kw in text for kw in keywords):
                detected.append(sector)
        
        return detected
    
    def classify_batch(self, articles: List[Dict[str, str]]) -> List[NewsClassification]:
        """
        Classify multiple articles
        
        Args:
            articles: List of dicts with 'title' and optionally 'content'
            
        Returns:
            List of NewsClassification objects
        """
        results = []
        
        for article in articles:
            title = article.get('title', '')
            content = article.get('content')
            
            if title:
                classification = self.classify(title, content)
                results.append(classification)
        
        return results
    
    def filter_relevant_news(self, 
                            articles: List[Dict[str, str]],
                            min_relevance: float = 0.4) -> List[NewsClassification]:
        """
        Filter news articles by relevance threshold
        
        Args:
            articles: List of articles
            min_relevance: Minimum relevance score (0-1)
            
        Returns:
            List of relevant NewsClassification objects
        """
        classifications = self.classify_batch(articles)
        
        return [
            c for c in classifications
            if c.relevance_score >= min_relevance
        ]
    
    def get_topic_summary(self, classifications: List[NewsClassification]) -> Dict:
        """
        Get summary of topics across multiple articles
        
        Args:
            classifications: List of NewsClassification objects
            
        Returns:
            Topic summary statistics
        """
        if not classifications:
            return {}
        
        all_topics = []
        all_regions = []
        all_sectors = []
        impact_counts = Counter()
        
        for c in classifications:
            all_topics.extend(c.topics)
            all_regions.extend(c.regions)
            all_sectors.extend(c.sectors)
            impact_counts[c.market_impact] += 1
        
        topic_counts = Counter(all_topics)
        region_counts = Counter(all_regions)
        sector_counts = Counter(all_sectors)
        
        return {
            'total_articles': len(classifications),
            'average_relevance': np.mean([c.relevance_score for c in classifications]),
            'top_topics': topic_counts.most_common(5),
            'top_regions': region_counts.most_common(5),
            'top_sectors': sector_counts.most_common(5),
            'impact_distribution': dict(impact_counts),
            'high_impact_count': impact_counts['high'],
            'medium_impact_count': impact_counts['medium'],
            'low_impact_count': impact_counts['low']
        }
    
    def get_trending_topics(self, 
                          classifications: List[NewsClassification],
                          time_window_hours: int = 24) -> List[Tuple[str, int]]:
        """
        Get trending topics in recent time window
        
        Args:
            classifications: List of classifications
            time_window_hours: Hours to look back
            
        Returns:
            List of (topic, count) tuples
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(hours=time_window_hours)
        
        recent = [
            c for c in classifications
            if c.timestamp > cutoff
        ]
        
        all_topics = []
        for c in recent:
            all_topics.extend(c.topics)
        
        topic_counts = Counter(all_topics)
        
        return topic_counts.most_common(10)


class NewsFilter:
    """Filter news by various criteria"""
    
    def __init__(self, classifier: NewsClassifier):
        """Initialize news filter"""
        self.classifier = classifier
    
    def filter_by_topic(self, 
                       articles: List[Dict[str, str]],
                       topics: List[str]) -> List[NewsClassification]:
        """Filter articles by specific topics"""
        classifications = self.classifier.classify_batch(articles)
        
        return [
            c for c in classifications
            if any(topic in c.topics for topic in topics)
        ]
    
    def filter_by_region(self,
                        articles: List[Dict[str, str]],
                        regions: List[str]) -> List[NewsClassification]:
        """Filter articles by region"""
        classifications = self.classifier.classify_batch(articles)
        
        return [
            c for c in classifications
            if any(region in c.regions for region in regions)
        ]
    
    def filter_high_impact(self,
                          articles: List[Dict[str, str]]) -> List[NewsClassification]:
        """Filter only high-impact articles"""
        classifications = self.classifier.classify_batch(articles)
        
        return [
            c for c in classifications
            if c.market_impact == 'high'
        ]
    
    def filter_german_stimulus(self,
                              articles: List[Dict[str, str]]) -> List[NewsClassification]:
        """Filter articles about German stimulus (key JPMorgan theme)"""
        return self.filter_by_topic(articles, ['german_stimulus', 'fiscal_policy'])
    
    def filter_defense_sector(self,
                             articles: List[Dict[str, str]]) -> List[NewsClassification]:
        """Filter defense sector news (JPMorgan sector winner)"""
        return self.filter_by_topic(articles, ['defense_spending'])