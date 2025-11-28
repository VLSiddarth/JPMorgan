"""
src/data/processors/validator.py

Data validation utilities for market, macro and portfolio time series.

This module is part of the DATA PROCESSING LAYER and is responsible for:
- Schema / type checks
- Basic data quality checks (missingness, duplicates, outliers)
- Producing structured validation reports that upstream components can use
  for alerting, logging or rejecting bad data.

It is intentionally conservative: it *does not* mutate the data aggressively.
Cleaning and transformations should live in `cleaner.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import logging
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class IssueSeverity(str, Enum):
    """Severity classification for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationIssue(BaseModel):
    """
    A single data validation issue.

    Attributes:
        code: Machine-readable code (e.g., "MISSING_DATA_PCT", "OUTLIER_DETECTED").
        severity: INFO / WARNING / ERROR.
        message: Human-readable explanation.
        context: Optional small payload with extra details
                 (percentages, counts, ranges, etc.).
    """

    code: str
    severity: IssueSeverity
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """
    Result of running validations on a dataset.

    Attributes:
        is_valid: Whether the dataset passed all *error-level* checks.
        issues: List of issues encountered (INFO/WARNING/ERROR).
        stats: Summary statistics about the validated data.
    """

    is_valid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    @property
    def infos(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.INFO]


@dataclass
class SeriesValidationConfig:
    """
    Configuration for validating a univariate time series.

    Attributes:
        min_points: Minimum acceptable number of non-null observations.
        max_missing_pct: Maximum allowed percentage of missing values [0, 100].
        outlier_std_threshold: Number of standard deviations for outlier flagging.
        enforce_datetime_index: Whether index must be datetime-like and monotonic.
        min_value: Optional lower bound for values.
        max_value: Optional upper bound for values.
    """

    min_points: int = settings.MIN_DATA_POINTS
    max_missing_pct: float = settings.MAX_MISSING_DATA_PCT
    outlier_std_threshold: float = settings.OUTLIER_STD_THRESHOLD
    enforce_datetime_index: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None


# ---------------------------------------------------------------------------
# Core Validator
# ---------------------------------------------------------------------------


class DataValidator:
    """
    Enterprise-grade data validator for time series and tabular data.

    Typical use:
        validator = DataValidator()
        report = validator.validate_price_series(price_series, symbol="^STOXX50E")
        if not report.is_valid:
            # log or reject dataset
    """

    def __init__(self, default_config: Optional[SeriesValidationConfig] = None) -> None:
        self.default_config = default_config or SeriesValidationConfig()

    # ------------------------
    # Public API
    # ------------------------

    def validate_price_series(
        self,
        series: pd.Series,
        symbol: str,
        config: Optional[SeriesValidationConfig] = None,
    ) -> ValidationReport:
        """
        Validate a price time series (e.g., close prices).

        Checks:
        - Index is datetime & monotonic (optional)
        - Minimum number of points
        - Percentage of missing values
        - Duplicate timestamps
        - Outlier values (z-score based)
        - Optional min/max value bounds

        Args:
            series: pandas Series with time index and numeric values.
            symbol: Identifier (ticker/index name) for logging / context.
            config: Optional SeriesValidationConfig (uses default if None).

        Returns:
            ValidationReport describing issues and stats.
        """
        cfg = config or self.default_config
        issues: List[ValidationIssue] = []
        s = series.copy()

        if s.empty:
            msg = f"Series for symbol '{symbol}' is empty."
            logger.error(msg)
            issues.append(
                ValidationIssue(
                    code="EMPTY_SERIES",
                    severity=IssueSeverity.ERROR,
                    message=msg,
                )
            )
            return ValidationReport(is_valid=False, issues=issues, stats={})

        # Ensure index is sorted
        if not s.index.is_monotonic_increasing:
            s = s.sort_index()
            issues.append(
                ValidationIssue(
                    code="INDEX_NOT_MONOTONIC",
                    severity=IssueSeverity.WARNING,
                    message=f"Index for '{symbol}' was not monotonic; sorted ascending.",
                )
            )

        # 1) Index checks
        if cfg.enforce_datetime_index:
            if not pd.api.types.is_datetime64_any_dtype(s.index):
                try:
                    s.index = pd.to_datetime(s.index)
                    issues.append(
                        ValidationIssue(
                            code="INDEX_CAST_TO_DATETIME",
                            severity=IssueSeverity.WARNING,
                            message=f"Index for '{symbol}' was cast to datetime.",
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    msg = (
                        f"Index for '{symbol}' is not datetime-like and cannot be cast. "
                        f"Error: {exc}"
                    )
                    logger.error(msg)
                    issues.append(
                        ValidationIssue(
                            code="INVALID_INDEX_TYPE",
                            severity=IssueSeverity.ERROR,
                            message=msg,
                        )
                    )
                    return ValidationReport(
                        is_valid=False,
                        issues=issues,
                        stats={"n": int(len(s))},
                    )

        # 2) Duplicate index
        if s.index.has_duplicates:
            dup_count = int(s.index.duplicated().sum())
            msg = f"Series '{symbol}' has {dup_count} duplicate timestamps."
            logger.warning(msg)
            issues.append(
                ValidationIssue(
                    code="DUPLICATE_INDEX",
                    severity=IssueSeverity.ERROR,
                    message=msg,
                    context={"duplicate_count": dup_count},
                )
            )

        # 3) Missing data
        non_null = s.notna().sum()
        total = len(s)
        missing = total - non_null
        missing_pct = (missing / total) * 100 if total > 0 else 0.0

        if non_null < cfg.min_points:
            msg = (
                f"Series '{symbol}' has only {non_null} non-null points "
                f"(min required: {cfg.min_points})."
            )
            logger.error(msg)
            issues.append(
                ValidationIssue(
                    code="INSUFFICIENT_DATA_POINTS",
                    severity=IssueSeverity.ERROR,
                    message=msg,
                    context={"non_null": int(non_null), "min_required": cfg.min_points},
                )
            )

        if missing_pct > cfg.max_missing_pct:
            msg = (
                f"Series '{symbol}' has {missing_pct:.2f}% missing data "
                f"(max allowed: {cfg.max_missing_pct:.2f}%)."
            )
            severity = (
                IssueSeverity.ERROR
                if missing_pct > cfg.max_missing_pct * 2
                else IssueSeverity.WARNING
            )
            logger.warning(msg)
            issues.append(
                ValidationIssue(
                    code="MISSING_DATA_PCT",
                    severity=severity,
                    message=msg,
                    context={
                        "missing_pct": float(missing_pct),
                        "missing_count": int(missing),
                        "total": int(total),
                    },
                )
            )

        # 4) Outlier detection (z-score based)
        outlier_stats = self._detect_outliers(s, cfg.outlier_std_threshold)
        if outlier_stats["count"] > 0:
            msg = (
                f"Series '{symbol}' has {outlier_stats['count']} potential outliers "
                f"(>|{cfg.outlier_std_threshold}| std dev from mean)."
            )
            logger.warning(msg)
            issues.append(
                ValidationIssue(
                    code="OUTLIERS_DETECTED",
                    severity=IssueSeverity.WARNING,
                    message=msg,
                    context=outlier_stats,
                )
            )

        # 5) Value range checks (optional)
        if cfg.min_value is not None:
            below_min = int((s < cfg.min_value).sum())
            if below_min > 0:
                msg = (
                    f"Series '{symbol}' has {below_min} values below min_value "
                    f"{cfg.min_value}."
                )
                logger.warning(msg)
                issues.append(
                    ValidationIssue(
                        code="VALUES_BELOW_MIN",
                        severity=IssueSeverity.ERROR,
                        message=msg,
                        context={"min_value": cfg.min_value, "count": below_min},
                    )
                )

        if cfg.max_value is not None:
            above_max = int((s > cfg.max_value).sum())
            if above_max > 0:
                msg = (
                    f"Series '{symbol}' has {above_max} values above max_value "
                    f"{cfg.max_value}."
                )
                logger.warning(msg)
                issues.append(
                    ValidationIssue(
                        code="VALUES_ABOVE_MAX",
                        severity=IssueSeverity.ERROR,
                        message=msg,
                        context={"max_value": cfg.max_value, "count": above_max},
                    )
                )

        is_valid = not any(i.severity == IssueSeverity.ERROR for i in issues)

        stats = {
            "symbol": symbol,
            "n_total": int(total),
            "n_non_null": int(non_null),
            "missing_pct": float(missing_pct),
            "min": float(np.nanmin(s.values)) if non_null > 0 else np.nan,
            "max": float(np.nanmax(s.values)) if non_null > 0 else np.nan,
            "mean": float(np.nanmean(s.values)) if non_null > 0 else np.nan,
            "std": float(np.nanstd(s.values)) if non_null > 0 else np.nan,
        }
        stats.update({f"outlier_{k}": v for k, v in outlier_stats.items()})

        if is_valid:
            logger.info(
                "Validation passed for symbol '%s' (n=%d, missing=%.2f%%).",
                symbol,
                total,
                missing_pct,
            )
        else:
            logger.error(
                "Validation FAILED for symbol '%s'. Errors: %d, Warnings: %d.",
                symbol,
                len([i for i in issues if i.severity == IssueSeverity.ERROR]),
                len([i for i in issues if i.severity == IssueSeverity.WARNING]),
            )

        return ValidationReport(is_valid=is_valid, issues=issues, stats=stats)

    def validate_dataframe_columns(
        self,
        df: pd.DataFrame,
        required_columns: Sequence[str],
        name: str = "DataFrame",
    ) -> ValidationReport:
        """
        Validate that all required columns exist and have valid data.

        This is a lightweight check for tabular datasets (e.g. factor panels).

        Args:
            df: Input DataFrame.
            required_columns: Columns that must be present.
            name: Logical name for logging (e.g., "factor_exposures").

        Returns:
            ValidationReport with column-level issues.
        """
        issues: List[ValidationIssue] = []

        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            msg = f"{name} missing required columns: {missing_cols}"
            logger.error(msg)
            issues.append(
                ValidationIssue(
                    code="MISSING_COLUMNS",
                    severity=IssueSeverity.ERROR,
                    message=msg,
                    context={"missing_columns": missing_cols},
                )
            )

        # If the structure is wrong, we stop here
        if missing_cols:
            return ValidationReport(
                is_valid=False,
                issues=issues,
                stats={"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])},
            )

        # Null checks per column
        null_stats: Dict[str, Dict[str, float]] = {}
        for col in required_columns:
            col_nulls = int(df[col].isna().sum())
            col_total = int(len(df[col]))
            col_missing_pct = (col_nulls / col_total) * 100 if col_total > 0 else 0.0
            null_stats[col] = {
                "missing_count": col_nulls,
                "missing_pct": float(col_missing_pct),
            }

            if col_missing_pct > settings.MAX_MISSING_DATA_PCT:
                msg = (
                    f"Column '{col}' in {name} has {col_missing_pct:.2f}% missing "
                    f"(max allowed: {settings.MAX_MISSING_DATA_PCT:.2f}%)."
                )
                logger.warning(msg)
                issues.append(
                    ValidationIssue(
                        code="COLUMN_MISSING_DATA_PCT",
                        severity=IssueSeverity.WARNING,
                        message=msg,
                        context={
                            "column": col,
                            "missing_pct": float(col_missing_pct),
                            "missing_count": col_nulls,
                            "total": col_total,
                        },
                    )
                )

        is_valid = not any(i.severity == IssueSeverity.ERROR for i in issues)
        stats = {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "null_stats": null_stats,
        }

        if is_valid:
            logger.info(
                "DataFrame '%s' validation passed (rows=%d, cols=%d).",
                name,
                df.shape[0],
                df.shape[1],
            )
        else:
            logger.error(
                "DataFrame '%s' validation FAILED. Errors: %d, Warnings: %d.",
                name,
                len([i for i in issues if i.severity == IssueSeverity.ERROR]),
                len([i for i in issues if i.severity == IssueSeverity.WARNING]),
            )

        return ValidationReport(is_valid=is_valid, issues=issues, stats=stats)

    # ------------------------
    # Internal helpers
    # ------------------------

    @staticmethod
    def _detect_outliers(
        s: pd.Series,
        std_threshold: float,
    ) -> Dict[str, Any]:
        """
        Simple z-score based outlier detection.

        Args:
            s: Input Series (numeric).
            std_threshold: Values with |z| > threshold are flagged.

        Returns:
            Dict with count, indices of outliers, and basic stats.
        """
        clean = s.dropna()
        n = len(clean)
        if n == 0:
            return {"count": 0, "indices": [], "mean": np.nan, "std": np.nan}

        mu = float(clean.mean())
        sigma = float(clean.std(ddof=0))

        if sigma == 0 or np.isnan(sigma):
            return {"count": 0, "indices": [], "mean": mu, "std": sigma}

        z_scores = (clean - mu) / sigma
        mask = z_scores.abs() > std_threshold
        outlier_idx = list(clean.index[mask])
        count = int(mask.sum())

        return {
            "count": count,
            "indices": [i.isoformat() if isinstance(i, pd.Timestamp) else str(i) for i in outlier_idx],
            "mean": mu,
            "std": sigma,
            "threshold": std_threshold,
        }


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

# You can import this in other modules:
# from src.data.processors.validator import validator
validator = DataValidator()


if __name__ == "__main__":
    # Quick self-test with synthetic data
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = pd.Series(
        np.random.lognormal(mean=0.0, sigma=0.02, size=100).cumprod() * 100,
        index=idx,
        name="TEST",
    )
    # Introduce some deliberate issues
    prices.iloc[5] = np.nan
    prices.iloc[10] = prices.iloc[10] * 20  # big outlier
    prices.index = list(prices.index[:-1]) + [prices.index[-2]]  # duplicate index

    rep = validator.validate_price_series(prices, symbol="TEST_INDEX")
    print("VALID:", rep.is_valid)
    print("ISSUES:")
    for issue in rep.issues:
        print(f"- [{issue.severity}] {issue.code}: {issue.message}")
    print("STATS:", rep.stats)
