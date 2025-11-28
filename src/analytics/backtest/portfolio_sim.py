# src/analytics/backtest/portfolio_sim.py

"""
Portfolio Simulation Engine
---------------------------

Provides a production-ready backtest / simulation engine for:

- Daily rebalanced portfolios based on target weights
- Incorporation of transaction costs via TransactionCostModel
- Generation of PortfolioPerformanceSeries

Inputs:
- Price history (DataFrame)
- Rebalancing schedule
- Target weights per rebalance date
- Transaction cost model

Outputs:
- PortfolioPerformanceSeries
- Turnover and transaction cost statistics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Callable

import numpy as np
import pandas as pd

from src.data.models.portfolio import (
    PortfolioPerformancePoint,
    PortfolioPerformanceSeries,
    PortfolioType,
    Currency,
)
from .transaction_cost import TransactionCostModel, TransactionCostParameters

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """
    Configuration parameters for the backtest.

    Attributes
    ----------
    initial_nav : float
        Starting NAV of the portfolio.
    base_currency : Currency | str
        Base currency of the portfolio.
    rebalance_frequency : str
        Pandas offset alias (e.g., 'M', 'W', 'Q') for scheduled rebalances.
    allow_short : bool
        Whether negative weights are allowed.
    max_leverage : float
        Maximum gross exposure (sum of absolute weights).
    """

    initial_nav: float = 100.0
    base_currency: Currency | str = Currency.EUR
    rebalance_frequency: str = "M"
    allow_short: bool = False
    max_leverage: float = 1.0


class PortfolioSimulator:
    """
    Simple but robust portfolio simulation engine.

    Assumptions:
    - Daily price data in DataFrame with columns = symbols
    - Rebalance on specified dates to given target weights
    - Between rebalances, weights drift with returns (no intra-period trading)
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        tc_model: Optional[TransactionCostModel] = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.tc_model = tc_model or TransactionCostModel()

    def run_backtest(
        self,
        prices: pd.DataFrame,
        target_weights: Dict[pd.Timestamp, pd.Series],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> PortfolioPerformanceSeries:
        """
        Run backtest given prices and dated target weights.

        Args:
            prices: DataFrame indexed by date, columns = symbols (prices)
            target_weights: dict[rebalance_date] -> Series of target weights
            benchmark_returns: optional Series of benchmark daily returns

        Returns:
            PortfolioPerformanceSeries
        """
        if prices is None or prices.empty:
            raise ValueError("Prices DataFrame cannot be empty")

        prices = prices.sort_index()
        dates = prices.index

        # Daily returns from prices
        rets = prices.pct_change().fillna(0.0)

        nav = self.config.initial_nav
        base_ccy = self.config.base_currency
        portfolio_id = "BACKTEST_PORTFOLIO"

        current_weights = pd.Series(0.0, index=prices.columns)
        performance_points: list[PortfolioPerformancePoint] = []

        # Sort target weights by date
        tw_dates = sorted(target_weights.keys())

        for i, dt in enumerate(dates):
            daily_ret = float((current_weights * rets.loc[dt]).sum())
            nav *= (1.0 + daily_ret)

            # Rebalance if dt in target_weights (or nearest after)
            if dt in target_weights:
                new_weights = target_weights[dt].reindex(prices.columns).fillna(0.0)

                # Enforce constraints
                new_weights = self._enforce_constraints(new_weights)

                # Transaction costs
                cost = self.tc_model.compute_transaction_costs(
                    prev_weights=current_weights,
                    new_weights=new_weights,
                    prices=prices.loc[dt],
                    portfolio_nav=nav,
                )
                nav += cost

                current_weights = new_weights

            bm_ret = float(benchmark_returns.loc[dt]) if benchmark_returns is not None and dt in benchmark_returns.index else None

            pf_point = PortfolioPerformancePoint(
                portfolio_id=portfolio_id,
                date=dt.date(),
                nav=nav,
                return_daily=daily_ret,
                benchmark_return_daily=bm_ret,
                gross_exposure=float(current_weights.abs().sum()),
                net_exposure=float(current_weights.sum()),
            )
            performance_points.append(pf_point)

        return PortfolioPerformanceSeries(
            portfolio_id=portfolio_id,
            points=performance_points,
        )

    def _enforce_constraints(self, weights: pd.Series) -> pd.Series:
        """
        Enforce constraints: no shorting (if disabled) and max leverage.
        """
        w = weights.copy()

        if not self.config.allow_short:
            w = w.clip(lower=0.0)

        gross = float(w.abs().sum())
        if gross == 0:
            return w

        # Scale to respect max_leverage
        if gross > self.config.max_leverage:
            scale = self.config.max_leverage / gross
            w *= scale

        # Renormalize to sum to 1 if leverage == 1
        if self.config.max_leverage == 1.0 and w.sum() != 0:
            w = w / w.sum()

        return w
