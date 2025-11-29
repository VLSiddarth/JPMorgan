"""
FinBERT Sentiment Analyzer

Institutional-grade financial sentiment module for the JPMorgan European
Equity Dashboard.

Features:
- Uses FinBERT (finance-specific BERT) via HuggingFace transformers.
- Supports batch inference for headlines / articles.
- Returns structured sentiment outputs:
    - label: positive / negative / neutral
    - confidence score
    - sentiment index (-100 to +100)
- Safe initialization (lazy model load, graceful fallback if transformers/torch missing).

Expected to be used by:
- src/analytics/sentiment/analyzer.py
- Notebooks for news & research sentiment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
except Exception:  # pragma: no cover - environment may not have transformers
    AutoTokenizer = None  # type: ignore
    AutoModelForSequenceClassification = None  # type: ignore
    pipeline = None  # type: ignore


@dataclass
class FinBertConfig:
    """
    Configuration for FinBERT sentiment analysis.

    Attributes:
        model_name: HuggingFace model identifier.
            Common choices:
                - "ProsusAI/finbert"
                - "yiyanghkust/finbert-tone"
        device: Device spec for transformers pipeline.
            - -1 -> CPU
            -  0 -> first GPU (if available)
        max_length: Max token length (truncation).
        batch_size: Batch size for pipeline inference.
        return_all_scores: Whether to return probabilities for all classes.
    """

    model_name: str = "ProsusAI/finbert"
    device: int = -1
    max_length: int = 128
    batch_size: int = 16
    return_all_scores: bool = True


class FinBertSentimentAnalyzer:
    """
    Wrapper around FinBERT model for financial sentiment analysis.

    Typical usage:

        analyzer = FinBertSentimentAnalyzer()
        df = analyzer.analyze_texts(
            ["ECB raises rates again", "European banks face new capital rules"]
        )

    Returned DataFrame columns:
        - text: Original text.
        - label: 'positive', 'negative', or 'neutral'.
        - score: Confidence score for predicted label (0-1).
        - sentiment_index: Scaled sentiment in [-100, +100].
        - positive_prob / neutral_prob / negative_prob: class probabilities (if available).
    """

    _PIPELINE_TASK = "sentiment-analysis"

    def __init__(self, config: Optional[FinBertConfig] = None) -> None:
        self.config: FinBertConfig = config or FinBertConfig()
        self._pipe = None  # Lazy-loaded transformers pipeline

        if AutoTokenizer is None or AutoModelForSequenceClassification is None or pipeline is None:
            logger.warning(
                "transformers library not available. FinBERT sentiment will NOT work "
                "until 'transformers' and 'torch' are installed."
            )

        logger.info(
            "FinBertSentimentAnalyzer initialized (model=%s, device=%d, max_length=%d)",
            self.config.model_name,
            self.config.device,
            self.config.max_length,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def analyze_texts(
        self,
        texts: Union[str, List[str]],
        metadata: Optional[Dict[str, List]] = None,
    ) -> pd.DataFrame:
        """
        Analyze sentiment for a single string or list of strings.

        Args:
            texts: Single text or list of texts (headlines, sentences, paragraphs).
            metadata: Optional additional columns to merge into the result.
                Example:
                    metadata = {"source": [...], "ticker": [...]}
                Length must match number of texts.

        Returns:
            DataFrame with sentiment results.
        """
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = [t for t in texts if isinstance(t, str) and t.strip()]

        if not texts_list:
            logger.warning("No valid texts provided to FinBERT analyzer.")
            return pd.DataFrame(columns=["text", "label", "score", "sentiment_index"])

        pipe = self._get_or_create_pipeline()
        if pipe is None:
            # transformers or torch missing
            logger.error("FinBERT pipeline not available. Returning empty DataFrame.")
            return pd.DataFrame(columns=["text", "label", "score", "sentiment_index"])

        try:
            outputs = pipe(
                texts_list,
                batch_size=self.config.batch_size,
                truncation=True,
                max_length=self.config.max_length,
            )
        except Exception as exc:
            logger.error("Error running FinBERT inference: %s", exc, exc_info=True)
            return pd.DataFrame(columns=["text", "label", "score", "sentiment_index"])

        # Normalize outputs to list-of-dicts with label+score etc.
        results_df = self._parse_outputs(texts_list, outputs)

        # Attach metadata if provided
        if metadata:
            meta_df = pd.DataFrame(metadata)
            if len(meta_df) != len(results_df):
                logger.warning(
                    "Metadata length (%d) does not match texts length (%d). "
                    "Metadata will be ignored.",
                    len(meta_df),
                    len(results_df),
                )
            else:
                results_df = pd.concat([results_df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

        return results_df

    def analyze_headlines(self, headlines: List[str], tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convenience wrapper specialized for headlines.

        Args:
            headlines: List of news headlines.
            tickers: Optional list of matching tickers for each headline.

        Returns:
            DataFrame with sentiment results and optional ticker column.
        """
        metadata: Dict[str, List] = {}
        if tickers is not None:
            if len(tickers) == len(headlines):
                metadata["ticker"] = tickers
            else:
                logger.warning(
                    "Length of tickers (%d) does not match headlines (%d). Ignoring ticker metadata.",
                    len(tickers),
                    len(headlines),
                )

        return self.analyze_texts(headlines, metadata=metadata)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_or_create_pipeline(self):
        """
        Lazily create the transformers pipeline.

        Returns:
            transformers.Pipeline or None if unavailable.
        """
        if self._pipe is not None:
            return self._pipe

        if AutoTokenizer is None or AutoModelForSequenceClassification is None or pipeline is None:
            logger.error(
                "transformers or torch library not available. "
                "Install them to enable FinBERT sentiment."
            )
            return None

        try:
            logger.info("Loading FinBERT model '%s' (this may take a while on first run)...", self.config.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name)

            self._pipe = pipeline(
                task=self._PIPELINE_TASK,
                model=model,
                tokenizer=tokenizer,
                device=self.config.device,
                return_all_scores=self.config.return_all_scores,
            )
            logger.info("FinBERT pipeline loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load FinBERT model '%s': %s", self.config.model_name, exc, exc_info=True)
            self._pipe = None

        return self._pipe

    def _parse_outputs(self, texts: List[str], outputs) -> pd.DataFrame:
        """
        Convert raw pipeline outputs into a clean DataFrame.

        FinBERT models often use labels like 'positive', 'negative', 'neutral',
        but some variants might have uppercase or prefixed labels. We normalize them.

        Args:
            texts: Original input texts.
            outputs: Raw pipeline outputs.

        Returns:
            DataFrame with fields:
                text, label, score, sentiment_index, positive_prob, neutral_prob, negative_prob
        """
        rows: List[Dict] = []

        # Some models with return_all_scores=True return:
        #   [ [ {'label':'positive','score':0.7}, {'label':'negative','score':0.2}, ... ], ... ]
        # Some with return_all_scores=False return:
        #   [ {'label':'positive','score':0.7}, ... ]
        for text, out in zip(texts, outputs):
            if isinstance(out, list):
                # list of scores for all labels
                label_scores = {self._normalize_label(d["label"]): float(d["score"]) for d in out}
                # pick best label
                best_label = max(label_scores.items(), key=lambda x: x[1])[0] if label_scores else "neutral"
                best_score = label_scores.get(best_label, np.nan)
            elif isinstance(out, dict):
                best_label = self._normalize_label(out.get("label", "neutral"))
                best_score = float(out.get("score", np.nan))
                label_scores = {best_label: best_score}
            else:
                logger.warning("Unexpected FinBERT output type: %r", type(out))
                best_label, best_score, label_scores = "neutral", np.nan, {}

            pos_prob = label_scores.get("positive", np.nan)
            neg_prob = label_scores.get("negative", np.nan)
            neu_prob = label_scores.get("neutral", np.nan)

            sentiment_index = self._compute_sentiment_index(
                positive_prob=pos_prob,
                negative_prob=neg_prob,
                neutral_prob=neu_prob,
                default_label=best_label,
            )

            rows.append(
                {
                    "text": text,
                    "label": best_label,
                    "score": best_score,
                    "sentiment_index": sentiment_index,
                    "positive_prob": pos_prob,
                    "neutral_prob": neu_prob,
                    "negative_prob": neg_prob,
                }
            )

        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def _normalize_label(label: str) -> str:
        """
        Map various label formats to standardized 'positive', 'negative', 'neutral'.

        Examples:
            'POSITIVE', 'LABEL_2' -> 'positive' (if known mapping)
        """
        if not label:
            return "neutral"

        l = label.strip().lower()

        # Common FinBERT label names
        if "pos" in l:
            return "positive"
        if "neg" in l:
            return "negative"
        if "neu" in l:
            return "neutral"

        # Generic sentiment-analysis models: LABEL_0, LABEL_1, LABEL_2
        # Often: 0=negative, 1=neutral, 2=positive (but may vary).
        if l in ("label_0", "0"):
            return "negative"
        if l in ("label_1", "1"):
            return "neutral"
        if l in ("label_2", "2"):
            return "positive"

        # Fallback
        return l

    @staticmethod
    def _compute_sentiment_index(
        positive_prob: float,
        negative_prob: float,
        neutral_prob: float,
        default_label: str,
    ) -> float:
        """
        Compute a sentiment index in [-100, +100].

        Logic:
        - If probabilities are available: (pos - neg) * 100
        - Else fall back to:
            +60 for positive
            -60 for negative
             0  for neutral

        Args:
            positive_prob: Probability for positive class.
            negative_prob: Probability for negative class.
            neutral_prob: Probability for neutral class (unused directly).
            default_label: Label used if probabilities are missing.

        Returns:
            sentiment_index in [-100, +100].
        """
        if not np.isnan(positive_prob) and not np.isnan(negative_prob):
            index = float((positive_prob - negative_prob) * 100.0)
            return max(min(index, 100.0), -100.0)

        # Fallback to label-only
        if default_label == "positive":
            return 60.0
        if default_label == "negative":
            return -60.0
        return 0.0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyzer = FinBertSentimentAnalyzer()

    sample_headlines = [
        "European banks rally as ECB signals end to rate hikes",
        "German industrial output disappoints, raising recession fears",
        "EU markets trade flat ahead of key inflation data",
    ]

    df = analyzer.analyze_headlines(sample_headlines)
    print("\nFinBERT sentiment results:")
    print(df)
