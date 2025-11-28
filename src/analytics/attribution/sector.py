# src/analytics/attribution/sector.py

"""
Sector Attribution Engine
-------------------------

Provides higher-level attribution utilities focusing on sectors:

- Sector return & contribution breakdown
- Sector over/underweight vs benchmark
- Simple sector attribution summary for CIO dashboards

Builds on top of:
- Portfolio/benchmark weights per asset
- Sector classification per asset
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .returns import ReturnAttributionEngine, AttributionResult

logger = logging.getLogger(__name__)


class SectorAttributionEngine:
    """
    Wrapper around ReturnAttributionEngine specialized for sectors.
    """

    def __init__(self) -> None:
        self._returns_engine = ReturnAttributionEngine()

    def attribute_by_sector(
        self,
        portfolio_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_weights: pd.Series,
        benchmark_returns: pd.Series,
        sector_mapping: pd.Series,
    ) -> AttributionResult:
        """
        Perform sector-level Brinson attribution.

        Args:
            portfolio_weights: weights per asset
            portfolio_returns: returns per asset
            benchmark_weights: benchmark weights per asset
            benchmark_returns: benchmark returns per asset
            sector_mapping: mapping asset -> sector (pd.Series)

        Returns:
            AttributionResult at sector level
        """
        return self._returns_engine.attribute_period(
            portfolio_weights=portfolio_weights,
            portfolio_returns=portfolio_returns,
            benchmark_weights=benchmark_weights,
            benchmark_returns=benchmark_returns,
            bucket_mapping=sector_mapping,
            bucket_type="sector",
        )

    def summary_table(
        self,
        result: AttributionResult,
    ) -> pd.DataFrame:
        """
        Build a compact sector attribution summary:

        Columns:
            ['allocation', 'selection', 'interaction', 'total']
        Index:
            sectors
        """
        df = pd.DataFrame(
            {
                "allocation": result.allocation_effect,
                "selection": result.selection_effect,
                "interaction": result.interaction_effect,
                "total": result.total_effect,
            }
        )
        return df.sort_values("total", ascending=False)

    def sector_exposure_table(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        sector_mapping: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute sector weights and over/underweights.

        Args:
            portfolio_weights: weights per asset
            benchmark_weights: weights per asset for benchmark
            sector_mapping: asset -> sector

        Returns:
            DataFrame with columns:
                ['portfolio_weight', 'benchmark_weight', 'active_weight']
        """
        pf_w, _ = portfolio_weights.align(sector_mapping, join="inner")
        bm_w, _ = benchmark_weights.align(sector_mapping, join="inner")
        sectors = sector_mapping.reindex(pf_w.index)

        pf_sector = pf_w.groupby(sectors).sum()
        bm_sector = bm_w.groupby(sectors).sum()

        idx = sorted(set(pf_sector.index).union(bm_sector.index))
        pf_sector = pf_sector.reindex(idx).fillna(0.0)
        bm_sector = bm_sector.reindex(idx).fillna(0.0)

        active = pf_sector - bm_sector
        df = pd.DataFrame(
            {
                "portfolio_weight": pf_sector,
                "benchmark_weight": bm_sector,
                "active_weight": active,
            }
        )
        return df.sort_values("active_weight", ascending=False)
