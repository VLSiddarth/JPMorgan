# src/analytics/attribution/returns.py

"""
Return Attribution Engine
-------------------------

Implements institutional-grade performance attribution:

- Arithmetic Brinson-style attribution:
  • Allocation effect
  • Selection effect
  • Interaction effect

- Multi-period aggregation of attribution effects

Inputs:
- Portfolio and benchmark weights/returns by asset or sector
- Daily (or periodic) return series

Outputs:
- DataFrames with contribution, allocation, selection, interaction
- Summary metrics for dashboards and reports
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """
    Container for single-period attribution results.

    Attributes
    ----------
    total_portfolio_return : float
        Observed portfolio return for the period.
    total_benchmark_return : float
        Observed benchmark return for the period.
    allocation_effect : pd.Series
        Allocation effect per bucket (e.g., sector or asset).
    selection_effect : pd.Series
        Selection effect per bucket.
    interaction_effect : pd.Series
        Interaction effect per bucket.
    total_effect : pd.Series
        Sum of allocation + selection + interaction per bucket.
    unexplained : float
        Difference between total portfolio excess return and sum of effects.
    """

    total_portfolio_return: float
    total_benchmark_return: float
    allocation_effect: pd.Series
    selection_effect: pd.Series
    interaction_effect: pd.Series
    total_effect: pd.Series
    unexplained: float


class ReturnAttributionEngine:
    """
    Arithmetic return attribution engine.

    Supports Brinson-style attribution on either:
    - Sector buckets (sector-level returns/weights)
    - Asset-level buckets (security-level returns/weights)

    The engine operates on cross-sectional data for a single period, and
    provides helpers to aggregate multi-period effects.

    Conventions:
    - All returns are simple returns for the given period (not log returns)
    - Weights are fractions of total portfolio/benchmark (sum to approx 1.0)
    """

    def __init__(self, check_weight_tolerance: float = 0.05) -> None:
        """
        Args:
            check_weight_tolerance:
                Tolerance for sum of weights deviating from 1.0. If weights
                sum outside [1 - tol, 1 + tol], a warning is logged.
        """
        self.check_weight_tolerance = check_weight_tolerance

    # --------------------------------------------------------------------- #
    # Core single-period attribution
    # --------------------------------------------------------------------- #
    def attribute_period(
        self,
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_weights: pd.Series,
        benchmark_returns: pd.Series,
        bucket_mapping: Optional[pd.Series] = None,
        bucket_type: Literal["sector", "asset"] = "asset",
    ) -> AttributionResult:
        """
        Perform Brinson-style attribution for a single period.

        Args:
            portfolio_weights:
                pd.Series indexed by asset (or bucket) with portfolio weights.
            portfolio_returns:
                pd.Series indexed by asset (or bucket) with portfolio returns.
            benchmark_weights:
                pd.Series indexed by asset (or bucket) with benchmark weights.
            benchmark_returns:
                pd.Series indexed by asset (or bucket) with benchmark returns.
            bucket_mapping:
                Optional mapping from asset -> bucket (e.g., ticker -> sector).
                If provided, attribution is aggregated by bucket.
            bucket_type:
                'sector' or 'asset' (just used for logging / metadata).

        Returns:
            AttributionResult
        """
        # Align all series
        pf_w, pf_r, bm_w, bm_r, bucket = self._align_inputs(
            portfolio_weights,
            portfolio_returns,
            benchmark_weights,
            benchmark_returns,
            bucket_mapping,
        )

        # Aggregate to bucket level if mapping provided
        if bucket is not None:
            # For weights: sum by bucket
            pf_w_b = pf_w.groupby(bucket).sum()
            bm_w_b = bm_w.groupby(bucket).sum()

            # For returns: compute weighted-average per bucket
            pf_r_b = (pf_r * pf_w).groupby(bucket).sum() / pf_w_b.replace(0, np.nan)
            bm_r_b = (bm_r * bm_w).groupby(bucket).sum() / bm_w_b.replace(0, np.nan)

            # Fill any NaNs (e.g., zero weights) with 0.0
            pf_r_b = pf_r_b.fillna(0.0)
            bm_r_b = bm_r_b.fillna(0.0)
            index = pf_w_b.index
        else:
            pf_w_b = pf_w
            bm_w_b = bm_w
            pf_r_b = pf_r
            bm_r_b = bm_r
            index = pf_w_b.index

        # Check weights
        self._check_weights(pf_w_b, label="portfolio")
        self._check_weights(bm_w_b, label="benchmark")

        # Total portfolio & benchmark returns
        rp = float((pf_w_b * pf_r_b).sum())
        rb = float((bm_w_b * bm_r_b).sum())

        # Brinson (arithmetic) decomposition:
        # Allocation: (w_p - w_b) * r_b
        allocation = (pf_w_b - bm_w_b) * bm_r_b

        # Selection: w_b * (r_p - r_b)
        selection = bm_w_b * (pf_r_b - bm_r_b)

        # Interaction: (w_p - w_b) * (r_p - r_b)
        interaction = (pf_w_b - bm_w_b) * (pf_r_b - bm_r_b)

        total_effect = allocation + selection + interaction

        # Excess return
        excess = rp - rb
        unexplained = float(excess - total_effect.sum())

        if abs(unexplained) > 1e-6:
            logger.warning(
                "Attribution unexplained component is %.6f (bucket_type=%s)",
                unexplained,
                bucket_type,
            )

        return AttributionResult(
            total_portfolio_return=rp,
            total_benchmark_return=rb,
            allocation_effect=allocation.reindex(index).fillna(0.0),
            selection_effect=selection.reindex(index).fillna(0.0),
            interaction_effect=interaction.reindex(index).fillna(0.0),
            total_effect=total_effect.reindex(index).fillna(0.0),
            unexplained=unexplained,
        )

    # --------------------------------------------------------------------- #
    # Multi-period aggregation
    # --------------------------------------------------------------------- #
    def aggregate_multi_period(
        self,
        period_results: Dict[pd.Timestamp, AttributionResult],
        method: Literal["arithmetic", "geometric"] = "arithmetic",
    ) -> pd.DataFrame:
        """
        Aggregate multiple AttributionResult objects into a summary table.

        Args:
            period_results:
                Dict mapping period end timestamp -> AttributionResult.
            method:
                'arithmetic' (simple sum) or 'geometric' (compounding) for
                portfolio/benchmark and excess returns. Effects are summed.

        Returns:
            pd.DataFrame with columns:
            ['allocation', 'selection', 'interaction', 'total', 'unexplained',
             'portfolio_return', 'benchmark_return', 'excess_return']
        """
        if not period_results:
            return pd.DataFrame()

        # Sort periods chronologically
        periods = sorted(period_results.keys())
        any_res = next(iter(period_results.values()))
        buckets = any_res.total_effect.index

        rows = []
        for t in periods:
            res = period_results[t]
            row = {
                "date": t,
                "allocation": res.allocation_effect,
                "selection": res.selection_effect,
                "interaction": res.interaction_effect,
                "total": res.total_effect,
                "unexplained": res.unexplained,
                "portfolio_return": res.total_portfolio_return,
                "benchmark_return": res.total_benchmark_return,
            }
            rows.append(row)

        # Build DataFrames for effects (bucket-level)
        alloc_df = pd.DataFrame(
            [r["allocation"] for r in rows],
            index=periods,
        ).reindex(columns=buckets)
        sel_df = pd.DataFrame(
            [r["selection"] for r in rows],
            index=periods,
        ).reindex(columns=buckets)
        int_df = pd.DataFrame(
            [r["interaction"] for r in rows],
            index=periods,
        ).reindex(columns=buckets)
        total_df = pd.DataFrame(
            [r["total"] for r in rows],
            index=periods,
        ).reindex(columns=buckets)

        # Scalars per period
        unexplained = pd.Series(
            [r["unexplained"] for r in rows], index=periods, name="unexplained"
        )
        rp_series = pd.Series(
            [r["portfolio_return"] for r in rows], index=periods, name="rp"
        )
        rb_series = pd.Series(
            [r["benchmark_return"] for r in rows], index=periods, name="rb"
        )

        if method == "geometric":
            # Geometric compounding for returns
            rp_total = (1 + rp_series).prod() - 1
            rb_total = (1 + rb_series).prod() - 1
        else:
            # Arithmetic sum of returns (approximation)
            rp_total = rp_series.sum()
            rb_total = rb_series.sum()

        excess_total = rp_total - rb_total

        # Summed effects across periods
        alloc_total = alloc_df.sum()
        sel_total = sel_df.sum()
        int_total = int_df.sum()
        total_effect = total_df.sum()

        # Build summary table at bucket level
        summary = pd.DataFrame(
            {
                "allocation": alloc_total,
                "selection": sel_total,
                "interaction": int_total,
                "total": total_effect,
            }
        )
        summary["unexplained"] = unexplained.sum()
        summary["portfolio_return"] = rp_total
        summary["benchmark_return"] = rb_total
        summary["excess_return"] = excess_total

        return summary

    # --------------------------------------------------------------------- #
    # Exposure / contribution helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def compute_contribution(
        weights: pd.Series,
        returns: pd.Series,
    ) -> pd.Series:
        """
        Compute contribution to return for each bucket.

        Args:
            weights: weights per bucket
            returns: returns per bucket

        Returns:
            pd.Series: contribution per bucket
        """
        w, r = weights.align(returns, join="inner")
        return w * r

    # --------------------------------------------------------------------- #
    # Internal input handling
    # --------------------------------------------------------------------- #
    def _align_inputs(
        self,
        pf_w: pd.Series,
        pf_r: pd.Series,
        bm_w: pd.Series,
        bm_r: pd.Series,
        bucket_mapping: Optional[pd.Series],
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, Optional[pd.Series]]:
        """
        Align all series to the same index, optionally aligning bucket mapping.
        """
        # Align portfolio weights & returns
        pf_w, pf_r = pf_w.align(pf_r, join="inner")
        # Align benchmark
        bm_w, bm_r = bm_w.align(bm_r, join="inner")

        # Align across portfolio & benchmark
        common_idx = pf_w.index.intersection(bm_w.index)
        pf_w = pf_w.reindex(common_idx).fillna(0.0)
        pf_r = pf_r.reindex(common_idx).fillna(0.0)
        bm_w = bm_w.reindex(common_idx).fillna(0.0)
        bm_r = bm_r.reindex(common_idx).fillna(0.0)

        if bucket_mapping is not None:
            bucket_mapping = bucket_mapping.reindex(common_idx)
        return pf_w, pf_r, bm_w, bm_r, bucket_mapping

    def _check_weights(self, w: pd.Series, label: str) -> None:
        """Log a warning if weights deviate significantly from 1.0."""
        s = float(w.sum())
        if not (1 - self.check_weight_tolerance <= s <= 1 + self.check_weight_tolerance):
            logger.warning(
                "Weights for %s sum to %.4f (outside tolerance ±%.2f)",
                label,
                s,
                self.check_weight_tolerance,
            )
