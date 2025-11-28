# src/analytics/signals/momentum.py

"""
Momentum Signal Engine
----------------------

Computes industry-grade momentum signals used in
quant strategies and CIO dashboards.

Signals implemented:
- MOM_1M, MOM_3M, MOM_6M, MOM_12M
- Risk-adjusted momentum
- Volatility-scaled momentum

Inputs:
- HistoricalPriceSeries or pandas.DataFrame with prices
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.models.signals import (
    SignalPoint,
    SignalCategory,
    SignalDirection,
    SignalStrength,
)

logger = logging.getLogger(__name__)


class MomentumSignalEngine:
    """
    Production-grade momentum signal engine used for:

    - Ranking securities
    - CIO dashboards (direction, strength)
    - Backtesting signals
    """

    LOOKBACK_WINDOWS = {
        "MOM_1M": 21,
        "MOM_3M": 63,
        "MOM_6M": 126,
        "MOM_12M": 252,
    }

    def compute_momentum_signals(
        self,
        df: pd.DataFrame,
        as_of: pd.Timestamp,
        symbol: str,
    ) -> Dict[str, SignalPoint]:
        """
        Compute all momentum signals for a single symbol.

        Args:
            df: DataFrame with 'close' prices, indexed by datetime
            as_of: timestamp as of which signals are computed
            symbol: ticker symbol

        Returns:
            dict: signal_name -> SignalPoint
        """
        if df is None or df.empty or "close" not in df.columns:
            logger.warning("Momentum engine received invalid DF for %s", symbol)
            return {}

        df = df.sort_index()
        signals: Dict[str, SignalPoint] = {}

        for name, window in self.LOOKBACK_WINDOWS.items():
            if len(df) < window + 1:
                continue

            try:
                past_price = df["close"].iloc[-window]
                current_price = df["close"].iloc[-1]

                momentum = (current_price / past_price) - 1.0
                direction, strength = self._classify_momentum(momentum)

                sp = SignalPoint(
                    symbol=symbol,
                    as_of=as_of,
                    category=SignalCategory.MOMENTUM,
                    name=name,
                    value=float(momentum),
                    direction=direction,
                    strength=strength,
                    lookback_days=window,
                )
                signals[name] = sp
            except Exception as e:
                logger.exception(
                    "Failed calculating %s momentum for %s: %s", name, symbol, e
                )

        # Add risk-adjusted momentum
        ram = self._compute_risk_adjusted_momentum(df, as_of, symbol)
        if ram:
            signals["MOM_RISK_ADJ"] = ram

        return signals

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    def _classify_momentum(
        self, value: float
    ) -> tuple[SignalDirection, SignalStrength]:
        """Convert numeric momentum into direction + strength."""
        if value >= 0.10:
            return SignalDirection.LONG, SignalStrength.STRONG_POSITIVE
        elif 0.03 <= value < 0.10:
            return SignalDirection.LONG, SignalStrength.POSITIVE
        elif -0.03 <= value < 0.03:
            return SignalDirection.NEUTRAL, SignalStrength.NEUTRAL
        elif -0.10 <= value < -0.03:
            return SignalDirection.SHORT, SignalStrength.NEGATIVE
        else:
            return SignalDirection.SHORT, SignalStrength.STRONG_NEGATIVE

    def _compute_risk_adjusted_momentum(
        self,
        df: pd.DataFrame,
        as_of: pd.Timestamp,
        symbol: str,
    ) -> Optional[SignalPoint]:
        """Compute risk-adjusted momentum (12M momentum / 3M volatility)."""
        try:
            if len(df) < 252:
                return None

            momentum_12m = (df["close"].iloc[-1] / df["close"].iloc[-252]) - 1
            vol_3m = df["close"].pct_change().iloc[-63:].std()

            if vol_3m == 0 or np.isnan(vol_3m):
                return None

            risk_adj = momentum_12m / vol_3m
            direction, strength = self._classify_momentum(momentum_12m)

            return SignalPoint(
                symbol=symbol,
                as_of=as_of,
                category=SignalCategory.MOMENTUM,
                name="MOM_RISK_ADJ",
                value=float(risk_adj),
                direction=direction,
                strength=strength,
                metadata={
                    "momentum_12m": float(momentum_12m),
                    "vol_3m": float(vol_3m),
                },
            )
        except Exception as e:
            logger.exception("Risk-adjusted momentum failed for %s: %s", symbol, e)
            return None
