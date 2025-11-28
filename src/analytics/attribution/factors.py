# src/analytics/attribution/factors.py

"""
Factor Attribution Engine
-------------------------

Computes factor-based performance attribution:

- Factor contribution based on:
  • Factor exposures (betas) per asset
  • Realized factor returns
  • Specific (idiosyncratic) returns

Supports:
- Cross-sectional factor contribution
- Aggregation to portfolio-level effects
- Residual (unexplained) component

Typical use-case:
- Fama-French Europe factors
- Custom JPMorgan factor models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FactorAttributionResult:
    """
    Container for factor attribution for a single period.

    Attributes
    ----------
    portfolio_return : float
        Observed portfolio return.
    benchmark_return : Optional[float]
        Optional benchmark return (can be used for relative attribution).
    factor_contributions : pd.Series
        Contribution per factor.
    specific_return : float
        Sum of idiosyncratic contributions.
    unexplained : float
        Difference between portfolio return and sum of factor + specific contributions.
    """

    portfolio_return: float
    benchmark_return: Optional[float]
    factor_contributions: pd.Series
    specific_return: float
    unexplained: float


class FactorAttributionEngine:
    """
    Factor-based attribution engine.

    Inputs:
        - weights: portfolio weights per asset
        - returns: realized asset returns
        - factor_exposures: DataFrame [asset x factor]
        - factor_returns: Series [factor] for the period

    Optionally:
        - benchmark_weights: weights for benchmark
    """

    def attribute_period(
        self,
        asset_weights: pd.Series,
        asset_returns: pd.Series,
        factor_exposures: pd.DataFrame,
        factor_returns: pd.Series,
        benchmark_weights: Optional[pd.Series] = None,
    ) -> FactorAttributionResult:
        """
        Perform factor attribution for a single period.

        Args:
            asset_weights: portfolio weights per asset
            asset_returns: realized asset returns for the period
            factor_exposures: DataFrame indexed by asset, columns = factor names
            factor_returns: Series indexed by factor, realized factor returns
            benchmark_weights: optional benchmark weights per asset

        Returns:
            FactorAttributionResult
        """
        # Align assets across inputs
        w, r = asset_weights.align(asset_returns, join="inner")
        X = factor_exposures.reindex(w.index).fillna(0.0)

        # Align factors
        X = X.loc[:, X.columns.intersection(factor_returns.index)]
        f = factor_returns.reindex(X.columns).fillna(0.0)

        if X.empty or w.empty or r.empty:
            logger.warning("FactorAttributionEngine: empty inputs for attribution.")
            return FactorAttributionResult(
                portfolio_return=0.0,
                benchmark_return=None,
                factor_contributions=pd.Series(dtype=float),
                specific_return=0.0,
                unexplained=0.0,
            )

        # Portfolio return
        rp = float((w * r).sum())

        # Factor contribution:
        # First compute portfolio factor exposures: sum_i (w_i * beta_i)
        pf_exposures = (X.mul(w, axis=0)).sum(axis=0)

        # Factor contributions: exposures * factor returns
        factor_contrib = pf_exposures * f

        # Predicted return from factors
        r_factor_pred = float(factor_contrib.sum())

        # Specific (idiosyncratic) return: residual at asset level
        # residual_i = r_i - (beta_i • f)
        beta_dot_f = (X * f.values).sum(axis=1)
        residuals = r - beta_dot_f
        specific_return = float((w * residuals).sum())

        total_explained = r_factor_pred + specific_return
        unexplained = float(rp - total_explained)

        # Optional benchmark return (not decomposed here)
        rb = None
        if benchmark_weights is not None and not benchmark_weights.empty:
            wb, _ = benchmark_weights.align(asset_returns, join="inner")
            rb = float((wb * r).sum())

        return FactorAttributionResult(
            portfolio_return=rp,
            benchmark_return=rb,
            factor_contributions=factor_contrib,
            specific_return=specific_return,
            unexplained=unexplained,
        )

    def to_dataframe(self, result: FactorAttributionResult) -> pd.DataFrame:
        """
        Convert attribution result to a DataFrame suitable for dashboards.

        Columns:
            ['contribution']
        Index:
            factor names + 'SPECIFIC' + 'UNEXPLAINED'
        """
        rows: Dict[str, float] = {
            factor: float(val)
            for factor, val in result.factor_contributions.items()
        }
        rows["SPECIFIC"] = result.specific_return
        rows["UNEXPLAINED"] = result.unexplained

        df = pd.DataFrame.from_dict(rows, orient="index", columns=["contribution"])
        return df.sort_values("contribution", ascending=False)
