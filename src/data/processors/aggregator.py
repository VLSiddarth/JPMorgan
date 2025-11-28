"""
src/data/processors/aggregator.py

Cross-source aggregation & reconciliation utilities.

Layer responsibilities:
- validator.py  → detect issues, no mutation
- cleaner.py    → clean individual series (fill, clip, etc.)
- normalizer.py → transform/scale series for modeling
- aggregator.py → combine multiple sources into coherent datasets

This module focuses on:
- Combining multiple providers for the same instrument (primary + fallbacks)
- Aligning market & macro data on a common calendar
- Building canonical panels (MultiIndex DataFrames) for analytics

Typical usage:

    from src.data.processors.validator import validator
    from src.data.processors.cleaner import cleaner
    from src.data.processors.normalizer import normalizer
    from src.data.processors.aggregator import aggregator, AggregationConfig

    primary = yahoo_loader.get_price_series("^STOXX50E")
    fallback = fred_loader.get_index_proxy("STOXX50E_FRED")

    merged, agg_report = aggregator.aggregate_time_series(
        symbol="^STOXX50E",
        primary=primary,
        fallbacks=[fallback],
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import logging
import numpy as np
import pandas as pd

from src.data.processors.validator import (
    validator,
    ValidationReport,
    IssueSeverity,
)
from src.data.processors.cleaner import cleaner
from src.data.processors.normalizer import normalizer, NormalizationConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AggregationAction(str, Enum):
    """High-level actions performed during aggregation."""

    ALIGN_INDICES = "align_indices"
    MERGE_PRIMARY_FALLBACK = "merge_primary_fallback"
    JOIN_MARKET_MACRO = "join_market_macro"
    BUILD_PANEL = "build_panel"
    RESAMPLE = "resample"
    FILL_GAPS = "fill_gaps"


@dataclass
class AggregationStepResult:
    """
    Single aggregation step outcome.

    Attributes:
        action: AggregationAction performed.
        description: Human-readable explanation.
        details: Structured metadata (row counts, dates, etc.).
    """

    action: AggregationAction
    description: str
    details: Dict[str, Any]


@dataclass
class AggregationReport:
    """
    Summary of aggregation operations for a dataset/symbol.

    Attributes:
        name: Logical dataset name or symbol.
        steps: List of step-level details.
        final_shape: Shape of resulting object (rows, cols).
    """

    name: str
    steps: List[AggregationStepResult]
    final_shape: Tuple[int, int]

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "final_shape": self.final_shape,
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
class AggregationConfig:
    """
    Configuration for cross-source time series aggregation.

    Attributes:
        resample_freq: Optional frequency string for resampling
                       (e.g. "B" for business daily, "M" for month-end).
        resample_method: "last", "mean", "sum" etc. for resample aggregation.
        gap_fill_method: "ffill", "bfill", "ffill_bfill", or "none".
        prefer_primary: If True, primary source always wins where it has data.
                        If False, allow fallbacks to overwrite based on recency.
    """

    resample_freq: Optional[str] = "B"
    resample_method: str = "last"
    gap_fill_method: str = "ffill_bfill"
    prefer_primary: bool = True


# ---------------------------------------------------------------------------
# Core Aggregator
# ---------------------------------------------------------------------------


class DataAggregator:
    """
    Enterprise-grade data aggregator for cross-source & cross-asset datasets.

    Main responsibilities:
    - Merge primary & fallback sources into a single canonical series
    - Align & join market and macro time series
    - Build multi-asset panels (e.g., sector baskets, factor portfolios)
    """

    def __init__(self, default_config: Optional[AggregationConfig] = None) -> None:
        self.default_config = default_config or AggregationConfig()

    # ------------------------------------------------------------------
    # 1. Primary + Fallback Aggregation
    # ------------------------------------------------------------------

    def aggregate_time_series(
        self,
        symbol: str,
        primary: pd.Series,
        fallbacks: Optional[List[pd.Series]] = None,
        config: Optional[AggregationConfig] = None,
    ) -> Tuple[pd.Series, AggregationReport]:
        """
        Merge primary and fallback time series into a single canonical series.

        Rules:
        - Index is unified across all sources (outer join).
        - For each timestamp:
            * Use primary if non-null.
            * Else, use first non-null fallback (in order).
        - Optionally resample to a standard frequency and fill small gaps.

        Args:
            symbol: Identifier for series.
            primary: Primary provider series.
            fallbacks: Ordered list of fallback provider series.
            config: AggregationConfig (optional).

        Returns:
            (merged_series, AggregationReport)
        """
        cfg = config or self.default_config
        fb_list = fallbacks or []
        steps: List[AggregationStepResult] = []

        # Ensure all series are sorted and aligned in type
        all_series = [primary] + fb_list
        aligned = []
        for idx, s in enumerate(all_series):
            name = "primary" if idx == 0 else f"fallback_{idx}"
            ser = s.copy()
            if not ser.index.is_monotonic_increasing:
                ser = ser.sort_index()
            # Cast to datetime index if possible
            if not pd.api.types.is_datetime64_any_dtype(ser.index):
                try:
                    ser.index = pd.to_datetime(ser.index)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(
                        "Failed to cast index to datetime for %s (%s): %s",
                        symbol,
                        name,
                        exc,
                    )
            aligned.append(ser)

        steps.append(
            AggregationStepResult(
                action=AggregationAction.ALIGN_INDICES,
                description="Aligned indices across primary and fallbacks.",
                details={
                    "sources": len(aligned),
                    "lengths": [int(len(s)) for s in aligned],
                },
            )
        )

        # Build union index
        union_index = aligned[0].index
        for ser in aligned[1:]:
            union_index = union_index.union(ser.index)
        union_index = union_index.sort_values()

        # Reindex all series to union
        reindexed = [s.reindex(union_index) for s in aligned]

        # Merge values: primary → first fallback → second fallback ...
        merged_values = reindexed[0].copy()
        for fb in reindexed[1:]:
            mask = merged_values.isna() & fb.notna()
            merged_values.loc[mask] = fb.loc[mask]

        # Log coverage
        non_null_primary = int(reindexed[0].notna().sum())
        non_null_total = int(merged_values.notna().sum())
        filled_by_fallback = non_null_total - non_null_primary

        steps.append(
            AggregationStepResult(
                action=AggregationAction.MERGE_PRIMARY_FALLBACK,
                description="Merged primary series with fallbacks using primary-first policy.",
                details={
                    "non_null_primary": non_null_primary,
                    "non_null_total": non_null_total,
                    "filled_by_fallback": int(filled_by_fallback),
                    "union_len": int(len(union_index)),
                },
            )
        )

        merged = merged_values

        # Optional resample
        if cfg.resample_freq is not None:
            before_len = len(merged)
            merged = self._resample_series(
                merged,
                freq=cfg.resample_freq,
                method=cfg.resample_method,
            )
            steps.append(
                AggregationStepResult(
                    action=AggregationAction.RESAMPLE,
                    description=(
                        f"Resampled series to frequency '{cfg.resample_freq}' "
                        f"using method '{cfg.resample_method}'."
                    ),
                    details={"before_len": int(before_len), "after_len": int(len(merged))},
                )
            )

        # Optional gap filling
        if cfg.gap_fill_method != "none":
            before_missing = int(merged.isna().sum())
            merged = self._fill_gaps(merged, method=cfg.gap_fill_method)
            after_missing = int(merged.isna().sum())

            steps.append(
                AggregationStepResult(
                    action=AggregationAction.FILL_GAPS,
                    description=f"Filled gaps using method '{cfg.gap_fill_method}'.",
                    details={
                        "missing_before": before_missing,
                        "missing_after": after_missing,
                    },
                )
            )

        final_shape = (len(merged), 1)
        report = AggregationReport(
            name=symbol,
            steps=steps,
            final_shape=final_shape,
        )

        logger.info(
            "Aggregation completed for '%s'. Steps=%d, len=%d",
            symbol,
            report.total_steps,
            len(merged),
        )
        return merged, report

    # ------------------------------------------------------------------
    # 2. Market + Macro Join
    # ------------------------------------------------------------------

    def join_market_and_macro(
        self,
        market_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        name: str = "market_macro",
        config: Optional[AggregationConfig] = None,
    ) -> Tuple[pd.DataFrame, AggregationReport]:
        """
        Join market data (equities) and macro data (FRED/ECB) on a common index.

        Steps:
        - Align indices (outer join).
        - Optionally resample to a standard frequency.
        - Gap-fill macro data with forward fill (typical for macro series).
        - Leave market series as-is except optional ffill for weekends/holidays.

        Args:
            market_df: DataFrame of market time series (prices, returns, etc.).
            macro_df: DataFrame of macro time series (GDP, PMI, yields, etc.).
            name: Dataset name for reporting.
            config: AggregationConfig.

        Returns:
            (combined_df, AggregationReport)
        """
        cfg = config or self.default_config
        steps: List[AggregationStepResult] = []

        # Align indices
        m = market_df.copy()
        mc = macro_df.copy()

        if not m.index.is_monotonic_increasing:
            m = m.sort_index()
        if not mc.index.is_monotonic_increasing:
            mc = mc.sort_index()

        # Cast both indices to datetime if needed
        for df, label in [(m, "market"), (mc, "macro")]:
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as exc:  # pragma: no cover
                    logger.error(
                        "Failed to cast index to datetime for '%s' df in '%s': %s",
                        label,
                        name,
                        exc,
                    )

        # Outer join
        combined = m.join(mc, how="outer", rsuffix="_macro")
        steps.append(
            AggregationStepResult(
                action=AggregationAction.JOIN_MARKET_MACRO,
                description="Joined market and macro data using outer join on index.",
                details={
                    "market_rows": int(m.shape[0]),
                    "macro_rows": int(mc.shape[0]),
                    "combined_rows": int(combined.shape[0]),
                    "market_cols": list(m.columns),
                    "macro_cols": list(mc.columns),
                },
            )
        )

        # Optional resample to business daily
        if cfg.resample_freq is not None:
            before_len = len(combined)
            combined = combined.resample(cfg.resample_freq).last()
            steps.append(
                AggregationStepResult(
                    action=AggregationAction.RESAMPLE,
                    description=(
                        f"Resampled combined market/macro data to '{cfg.resample_freq}' "
                        f"using 'last' aggregation."
                    ),
                    details={"before_len": int(before_len), "after_len": int(len(combined))},
                )
            )

        # Gap fill: macro forward-fill is standard, market ffill for weekends/gaps
        macro_cols = [c for c in combined.columns if c in mc.columns or c.endswith("_macro")]
        market_cols = [c for c in combined.columns if c not in macro_cols]

        before_missing_macro = int(combined[macro_cols].isna().sum().sum()) if macro_cols else 0
        before_missing_market = int(combined[market_cols].isna().sum().sum()) if market_cols else 0

        if macro_cols:
            combined[macro_cols] = combined[macro_cols].ffill()
        if market_cols:
            combined[market_cols] = combined[market_cols].ffill()

        after_missing_macro = int(combined[macro_cols].isna().sum().sum()) if macro_cols else 0
        after_missing_market = int(combined[market_cols].isna().sum().sum()) if market_cols else 0

        steps.append(
            AggregationStepResult(
                action=AggregationAction.FILL_GAPS,
                description="Forward-filled macro and market series after join.",
                details={
                    "macro_missing_before": before_missing_macro,
                    "macro_missing_after": after_missing_macro,
                    "market_missing_before": before_missing_market,
                    "market_missing_after": after_missing_market,
                },
            )
        )

        final_shape = combined.shape
        report = AggregationReport(
            name=name,
            steps=steps,
            final_shape=final_shape,
        )

        logger.info(
            "Market+macro join completed for '%s'. Steps=%d, shape=%s",
            name,
            report.total_steps,
            final_shape,
        )
        return combined, report

    # ------------------------------------------------------------------
    # 3. Panel Construction (e.g., baskets, factors)
    # ------------------------------------------------------------------

    def build_panel_from_series_dict(
        self,
        series_dict: Dict[str, pd.Series],
        name: str = "panel",
        normalize_returns: bool = False,
        normalization_config: Optional[NormalizationConfig] = None,
    ) -> Tuple[pd.DataFrame, AggregationReport]:
        """
        Build a MultiAsset panel from a dict of {symbol: Series}.

        Steps:
        - Align indices across all symbols (outer join).
        - Build a DataFrame with columns as symbols.
        - Optionally normalize each column to returns/z-scores.

        Args:
            series_dict: Mapping symbol → series.
            name: Panel name.
            normalize_returns: Whether to convert to (normalized) returns.
            normalization_config: Optional NormalizationConfig for columns.

        Returns:
            (panel_df, AggregationReport)
        """
        steps: List[AggregationStepResult] = []

        if not series_dict:
            logger.warning("build_panel_from_series_dict called with empty dict '%s'.", name)
            empty_df = pd.DataFrame()
            report = AggregationReport(
                name=name,
                steps=[],
                final_shape=empty_df.shape,
            )
            return empty_df, report

        # Align all indices via outer join
        union_index = None
        for symbol, series in series_dict.items():
            s = series.copy()
            if not s.index.is_monotonic_increasing:
                s = s.sort_index()
            if not pd.api.types.is_datetime64_any_dtype(s.index):
                try:
                    s.index = pd.to_datetime(s.index)
                except Exception as exc:  # pragma: no cover
                    logger.error(
                        "Failed to cast index to datetime for symbol '%s' in panel '%s': %s",
                        symbol,
                        name,
                        exc,
                    )
            if union_index is None:
                union_index = s.index
            else:
                union_index = union_index.union(s.index)

        union_index = union_index.sort_values()
        panel = pd.DataFrame(index=union_index)

        for symbol, series in series_dict.items():
            panel[symbol] = series.reindex(union_index)

        steps.append(
            AggregationStepResult(
                action=AggregationAction.BUILD_PANEL,
                description="Built panel DataFrame from multiple symbol series.",
                details={
                    "symbols": list(series_dict.keys()),
                    "rows": int(panel.shape[0]),
                    "cols": int(panel.shape[1]),
                },
            )
        )

        # Optional normalization (per column)
        if normalize_returns:
            cfg = normalization_config or NormalizationConfig(
                price_to_return=True,
                log_return=False,
                standardize="zscore",
                center=False,
                scale_by_vol=False,
            )
            norm_panel, norm_report = normalizer.normalize_dataframe(
                panel,
                name=name,
                config=cfg,
                per_column=True,
            )
            panel = norm_panel
            steps.append(
                AggregationStepResult(
                    action=AggregationAction.FILL_GAPS,
                    description="Applied normalization (returns + standardization) to panel.",
                    details={"normalization_steps": [s.action.value for s in norm_report.steps]},
                )
            )

        final_shape = panel.shape
        report = AggregationReport(
            name=name,
            steps=steps,
            final_shape=final_shape,
        )

        logger.info(
            "Panel construction completed for '%s'. Steps=%d, shape=%s",
            name,
            report.total_steps,
            final_shape,
        )
        return panel, report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_series(
        s: pd.Series,
        freq: str,
        method: str = "last",
    ) -> pd.Series:
        """Resample series to given frequency using chosen aggregation method."""
        agg_map = {
            "last": "last",
            "mean": "mean",
            "sum": "sum",
            "first": "first",
            "max": "max",
            "min": "min",
        }
        agg_method = agg_map.get(method, "last")
        return s.resample(freq).agg(agg_method)

    @staticmethod
    def _fill_gaps(
        s: pd.Series,
        method: str,
    ) -> pd.Series:
        """
        Fill gaps in a series using chosen method.

        method:
            - "ffill": forward fill
            - "bfill": backward fill
            - "ffill_bfill": forward then backward
        """
        if method == "ffill":
            return s.ffill()
        if method == "bfill":
            return s.bfill()
        if method == "ffill_bfill":
            return s.ffill().bfill()
        return s


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

# Import this where needed:
# from src.data.processors.aggregator import aggregator, AggregationConfig
aggregator = DataAggregator()


if __name__ == "__main__":
    # Simple self-test with synthetic data
    idx1 = pd.date_range("2024-01-01", periods=10, freq="B")
    idx2 = pd.date_range("2024-01-03", periods=8, freq="B")

    primary = pd.Series(np.arange(10, dtype=float), index=idx1)
    fallback = pd.Series(np.arange(100, 108, dtype=float), index=idx2)
    primary.iloc[3:5] = np.nan  # introduce gaps

    da = DataAggregator()
    merged, rep = da.aggregate_time_series(
        symbol="TEST",
        primary=primary,
        fallbacks=[fallback],
    )

    print("Merged series:")
    print(merged.head(15))
    print("\nAggregation steps:")
    for step in rep.steps:
        print(f"- {step.action.value}: {step.description} ({step.details})")
