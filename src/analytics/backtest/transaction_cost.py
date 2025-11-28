# src/analytics/backtest/transaction_cost.py

"""
Transaction Cost Model
----------------------

Provides production-ready functions to model:

- Explicit transaction costs:
  • Commissions
  • Fees
- Implicit costs:
  • Bid-ask spread
  • Market impact (simple parametric model)
- Slippage vs mid-price

Used by:
- Backtest engine
- Portfolio simulation
- Pre-trade analytics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransactionCostParameters:
    """
    Parameters for transaction cost modeling.

    Attributes
    ----------
    commission_bps : float
        Commission in basis points of notional (1 bp = 0.01%).
    spread_bps : float
        Bid-ask spread in basis points of price (round-trip).
    impact_coeff : float
        Market impact coefficient. Used as:
            impact = impact_coeff * (trade_size / ADV) ^ 0.5
    min_cost_bps : float
        Minimum cost floor in bps.
    """

    commission_bps: float = 1.0
    spread_bps: float = 5.0
    impact_coeff: float = 10.0
    min_cost_bps: float = 0.5


class TransactionCostModel:
    """
    Transaction cost model applied at asset level per rebalance.

    The model is simple but configurable, and can be replaced with more
    sophisticated models if needed.
    """

    def __init__(self, params: Optional[TransactionCostParameters] = None) -> None:
        self.params = params or TransactionCostParameters()

    def estimate_round_trip_cost_bps(
        self,
        turnover_pct: float,
        rel_size_vs_adv: Optional[float] = None,
    ) -> float:
        """
        Estimate round-trip cost in bps for a given turnover.

        Args:
            turnover_pct: fraction of portfolio traded (0.10 = 10%)
            rel_size_vs_adv: trade size / ADV (if known)

        Returns:
            Estimated cost in bps (round-trip).
        """
        p = self.params

        # Commission
        commission = p.commission_bps

        # Spread cost: half-spread per side
        spread = p.spread_bps

        # Impact (optional)
        impact = 0.0
        if rel_size_vs_adv is not None and rel_size_vs_adv > 0:
            impact = p.impact_coeff * np.sqrt(rel_size_vs_adv)

        total = max(commission + spread + impact, p.min_cost_bps)
        return float(total)

    def compute_transaction_costs(
        self,
        prev_weights: pd.Series,
        new_weights: pd.Series,
        prices: pd.Series,
        portfolio_nav: float,
        adv: Optional[pd.Series] = None,
    ) -> float:
        """
        Compute transaction costs in portfolio base currency for a rebalance.

        Args:
            prev_weights: current portfolio weights
            new_weights: target portfolio weights
            prices: latest prices per asset (in base currency or converted)
            portfolio_nav: portfolio NAV in base currency
            adv: average daily volume in notional (same currency), optional

        Returns:
            Transaction cost in base currency (negative number).
        """
        # Align
        prev_w, new_w = prev_weights.align(new_weights, join="outer", fill_value=0.0)
        prices = prices.reindex(prev_w.index).fillna(0.0)

        # Trade sizes in weight terms
        trade_weights = (new_w - prev_w).abs()

        if trade_weights.sum() == 0:
            return 0.0

        # Notional traded per asset
        trade_notional = trade_weights * portfolio_nav
        total_notional = float(trade_notional.sum())

        # Average relative size vs ADV for rough impact modeling
        avg_rel_size: Optional[float] = None
        if adv is not None:
            adv = adv.reindex(trade_notional.index).replace(0.0, np.nan)
            rel_size = (trade_notional / adv).dropna()
            if not rel_size.empty:
                avg_rel_size = float(rel_size.mean())

        total_cost_bps = self.estimate_round_trip_cost_bps(
            turnover_pct=float(trade_weights.sum()),
            rel_size_vs_adv=avg_rel_size,
        )

        # Cost in currency: total_notional * (bps / 10_000)
        cost_currency = -total_notional * (total_cost_bps / 10_000.0)
        return float(cost_currency)
