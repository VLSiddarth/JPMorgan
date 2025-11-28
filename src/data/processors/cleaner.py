"""
src/data/processors/cleaner.py

Data cleaning utilities for market, macro and portfolio time series.

This module follows the JPMorgan-style separation of concerns:

- validator.py  → Detect issues, do NOT mutate data
- cleaner.py    → Apply deterministic cleaning steps, log what changed
- normalizer.py → Scale/transform data for modeling/analytics
- aggregator.py → Cross-source reconciliation and aggregation

Typical usage:

    from src.data.processors.validator import validator
    from src.data.processors.cleaner import cleaner, SeriesCleaningConfig

    report = validator.validate_price_series(series, symbol="^STOXX50E")
    cleaned, cleanup_report = cleaner.clean_price_series(
        series,
        symbol="^STOXX50E",
        validation_report=report,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import logging
import numpy as np
import pandas as pd

from config.settings import settings
from src.data.processors.validator import (
    ValidationReport,
    ValidationIssue,
    IssueSeverity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CleaningAction(str, Enum):
    """Enumeration of supported cleaning actions."""

    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    CLIP_OUTLIERS = "clip_outliers"
    ENFORCE_BOUNDS = "enforce_bounds"
    DROP_DUPLICATES = "drop_duplicates"
    SORT_INDEX = "sort_index"
    CAST_INDEX_TO_DATETIME = "cast_index_to_datetime"


@dataclass
class CleaningStepResult:
    """
    A single cleaning step outcome.

    Attributes:
        action: CleaningAction performed.
        description: Human-readable summary of what was done.
        details: Optional structured metadata (counts, thresholds, etc.).
    """

    action: CleaningAction
    description: str
    details: Dict[str, Any]


@dataclass
class CleaningReport:
    """
    Summary of all cleaning operations applied to a series.

    Attributes:
        symbol: Series identifier (e.g., ticker).
        original_length: Original number of observations.
        cleaned_length: Number of observations after cleaning.
        steps: List of step-level results.
    """

    symbol: str
    original_length: int
    cleaned_length: int
    steps: List[CleaningStepResult]

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "original_length": self.original_length,
            "cleaned_length": self.cleaned_length,
            "total_steps": self.total_steps,
            "steps": [
                {
                    "action": step.action.value,
                    "description": step.description,
                    "details": step.details,
                }
                for step in self.steps
            ],
        }


@dataclass
class SeriesCleaningConfig:
    """
    Configuration for cleaning a univariate time series.

    Attributes:
        max_missing_pct_after: Target maximum missing percentage after cleaning.
        fill_strategy: How to fill missing values ("ffill", "bfill", "ffill_bfill",
                       "interpolate", or "none").
        interpolate_method: pandas interpolation method (e.g. "time", "linear").
        clip_outliers: Whether to clip outliers.
        outlier_std_threshold: Z-score threshold for clipping.
        min_value: Optional hard lower bound for values.
        max_value: Optional hard upper bound for values.
        drop_duplicates: Whether to drop duplicate timestamps.
        cast_index_to_datetime: Whether to cast index to datetime (if not already).
    """

    max_missing_pct_after: float = settings.MAX_MISSING_DATA_PCT
    fill_strategy: str = "ffill_bfill"  # "ffill", "bfill", "ffill_bfill", "interpolate", "none"
    interpolate_method: str = "time"
    clip_outliers: bool = True
    outlier_std_threshold: float = settings.OUTLIER_STD_THRESHOLD
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    drop_duplicates: bool = True
    cast_index_to_datetime: bool = True


# ---------------------------------------------------------------------------
# Core Cleaner
# ---------------------------------------------------------------------------


class DataCleaner:
    """
    Enterprise-grade cleaner for time series.

    Designed to be deterministic, transparent and compatible with the
    validation reports produced by `DataValidator`.

    High-level entry point:
        clean_price_series()
    """

    def __init__(self, default_config: Optional[SeriesCleaningConfig] = None) -> None:
        self.default_config = default_config or SeriesCleaningConfig()

    # ------------------------
    # Public API
    # ------------------------

    def clean_price_series(
        self,
        series: pd.Series,
        symbol: str,
        validation_report: Optional[ValidationReport] = None,
        config: Optional[SeriesCleaningConfig] = None,
    ) -> Tuple[pd.Series, CleaningReport]:
        """
        Clean a price time series using a sensible sequence of operations.

        Steps (high level):
        1. Sort index, optionally cast to datetime.
        2. Drop duplicates if configured.
        3. Fill missing values according to fill_strategy.
        4. Optionally interpolate remaining gaps.
        5. Optionally clip outliers via z-scores.
        6. Enforce optional min/max bounds.

        Args:
            series: Input Series (time-indexed, numeric).
            symbol: Identifier (e.g. ticker).
            validation_report: Optional report from validator (for context).
            config: Optional cleaning configuration.

        Returns:
            (cleaned_series, CleaningReport)
        """
        cfg = config or self.default_config
        s = series.copy()
        original_length = len(s)
        steps: List[CleaningStepResult] = []

        if s.empty:
            logger.warning("clean_price_series called with empty series for '%s'.", symbol)
            report = CleaningReport(
                symbol=symbol,
                original_length=original_length,
                cleaned_length=original_length,
                steps=[],
            )
            return s, report

        # 1) Sort index + cast index to datetime (optional)
        if not s.index.is_monotonic_increasing:
            s = s.sort_index()
            steps.append(
                CleaningStepResult(
                    action=CleaningAction.SORT_INDEX,
                    description="Sorted index ascending.",
                    details={"before_monotonic": False},
                )
            )

        if cfg.cast_index_to_datetime and not pd.api.types.is_datetime64_any_dtype(s.index):
            try:
                s.index = pd.to_datetime(s.index)
                steps.append(
                    CleaningStepResult(
                        action=CleaningAction.CAST_INDEX_TO_DATETIME,
                        description="Cast index to datetime.",
                        details={},
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to cast index to datetime for '%s': %s", symbol, exc)

        # 2) Drop duplicate timestamps
        if cfg.drop_duplicates and s.index.has_duplicates:
            before = len(s)
            s = s[~s.index.duplicated(keep="first")]
            removed = before - len(s)
            steps.append(
                CleaningStepResult(
                    action=CleaningAction.DROP_DUPLICATES,
                    description=f"Dropped {removed} duplicate timestamps.",
                    details={"removed": int(removed), "before": int(before), "after": int(len(s))},
                )
            )

        # 3) Fill missing values according to strategy
        s, fill_step = self._apply_fill_strategy(s, cfg.fill_strategy, cfg.interpolate_method)
        if fill_step is not None:
            steps.append(fill_step)

        # 4) Optional outlier clipping
        if cfg.clip_outliers:
            s, outlier_step = self._clip_outliers(s, cfg.outlier_std_threshold)
            if outlier_step is not None:
                steps.append(outlier_step)

        # 5) Enforce bounds
        s, bounds_step = self._enforce_bounds(s, cfg.min_value, cfg.max_value)
        if bounds_step is not None:
            steps.append(bounds_step)

        cleaned_length = len(s)
        report = CleaningReport(
            symbol=symbol,
            original_length=original_length,
            cleaned_length=cleaned_length,
            steps=steps,
        )

        # Log summary
        logger.info(
            "Cleaning completed for '%s'. Original len=%d, cleaned len=%d, steps=%d",
            symbol,
            original_length,
            cleaned_length,
            report.total_steps,
        )

        # Optional: If a validation report was provided and still has severe issues,
        # you might decide to re-run validation or log a combined summary here.
        if validation_report is not None and not validation_report.is_valid:
            logger.warning(
                "Cleaning applied to series '%s' that had validation errors. "
                "Consider re-validating post-cleaning.",
                symbol,
            )

        return s, report

    # ------------------------
    # Internal helpers
    # ------------------------

    @staticmethod
    def _apply_fill_strategy(
        s: pd.Series,
        strategy: str,
        interpolate_method: str,
    ) -> Tuple[pd.Series, Optional[CleaningStepResult]]:
        """
        Apply missing-value filling according to selected strategy.

        Supported strategies:
            - "ffill": forward fill
            - "bfill": backward fill
            - "ffill_bfill": forward fill then backward fill
            - "interpolate": pandas interpolate()
            - "none": do nothing
        """
        null_before = int(s.isna().sum())
        if null_before == 0 or strategy == "none":
            return s, None

        s_filled = s.copy()

        if strategy == "ffill":
            s_filled = s_filled.ffill()
            action = CleaningAction.FORWARD_FILL
            description = "Forward-filled missing values."

        elif strategy == "bfill":
            s_filled = s_filled.bfill()
            action = CleaningAction.BACKWARD_FILL
            description = "Backward-filled missing values."

        elif strategy == "ffill_bfill":
            s_filled = s_filled.ffill().bfill()
            action = CleaningAction.FORWARD_FILL
            description = "Forward-filled then backward-filled missing values."

        elif strategy == "interpolate":
            try:
                s_filled = s_filled.interpolate(method=interpolate_method, limit_direction="both")
                action = CleaningAction.INTERPOLATE
                description = f"Interpolated missing values (method='{interpolate_method}')."
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Interpolation failed: %s", exc)
                return s, None
        else:
            logger.warning("Unknown fill_strategy '%s'. No filling applied.", strategy)
            return s, None

        null_after = int(s_filled.isna().sum())
        return s_filled, CleaningStepResult(
            action=action,
            description=description,
            details={"missing_before": null_before, "missing_after": null_after},
        )

    @staticmethod
    def _clip_outliers(
        s: pd.Series,
        std_threshold: float,
    ) -> Tuple[pd.Series, Optional[CleaningStepResult]]:
        """
        Clip outliers based on z-score threshold.

        Values with |z| > threshold are clipped to the boundary value
        (mean ± threshold * std).

        Returns:
            (clipped_series, CleaningStepResult or None)
        """
        clean = s.dropna()
        if clean.empty:
            return s, None

        mu = float(clean.mean())
        sigma = float(clean.std(ddof=0))
        if sigma == 0 or np.isnan(sigma):
            return s, None

        upper = mu + std_threshold * sigma
        lower = mu - std_threshold * sigma

        before_outliers = int(((clean > upper) | (clean < lower)).sum())
        if before_outliers == 0:
            return s, None

        clipped = s.clip(lower=lower, upper=upper)
        after_outliers = int(((clipped > upper) | (clipped < lower)).sum())

        step = CleaningStepResult(
            action=CleaningAction.CLIP_OUTLIERS,
            description=(
                f"Clipped outliers using z-score threshold {std_threshold:.2f} "
                f"around mean={mu:.4f}, std={sigma:.4f}."
            ),
            details={
                "mean": mu,
                "std": sigma,
                "threshold": std_threshold,
                "bounds": {"lower": lower, "upper": upper},
                "outliers_before": before_outliers,
                "outliers_after": after_outliers,
            },
        )
        return clipped, step

    @staticmethod
    def _enforce_bounds(
        s: pd.Series,
        min_value: Optional[float],
        max_value: Optional[float],
    ) -> Tuple[pd.Series, Optional[CleaningStepResult]]:
        """
        Enforce hard min/max bounds if provided.

        Typical use cases:
        - Prices must be > 0
        - Rates cannot be below -100% or above 1000%, etc.
        """
        if min_value is None and max_value is None:
            return s, None

        below = int((s < min_value).sum()) if min_value is not None else 0
        above = int((s > max_value).sum()) if max_value is not None else 0

        if below == 0 and above == 0:
            return s, None

        clipped = s.copy()
        if min_value is not None:
            clipped[clipped < min_value] = min_value
        if max_value is not None:
            clipped[clipped > max_value] = max_value

        step = CleaningStepResult(
            action=CleaningAction.ENFORCE_BOUNDS,
            description="Enforced hard min/max bounds on series.",
            details={
                "min_value": min_value,
                "max_value": max_value,
                "values_below_min": below,
                "values_above_max": above,
            },
        )
        return clipped, step


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

# Import this where needed:
# from src.data.processors.cleaner import cleaner, SeriesCleaningConfig
cleaner = DataCleaner()


if __name__ == "__main__":
    # Quick self-test with synthetic data
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    s = pd.Series(
        np.random.lognormal(mean=0.0, sigma=0.02, size=100).cumprod() * 100,
        index=idx,
        name="TEST",
    )
    # Deliberate issues
    s.iloc[5:8] = np.nan
    s.iloc[10] = s.iloc[10] * 20  # big spike
    s.index = list(s.index[:-1]) + [s.index[-2]]  # duplicate timestamp

    from src.data.processors.validator import validator

    val_report = validator.validate_price_series(s, symbol="TEST_INDEX")
    cleaned, clean_report = cleaner.clean_price_series(
        s,
        symbol="TEST_INDEX",
        validation_report=val_report,
    )

    print("Original length:", len(s))
    print("Cleaned length :", len(cleaned))
    print("Cleaning steps :")
    for step in clean_report.steps:
        print(f"- {step.action.value}: {step.description} ({step.details})")
