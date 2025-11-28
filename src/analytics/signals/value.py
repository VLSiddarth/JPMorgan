# src/analytics/signals/value.py

"""
Value Signal Engine
-------------------

Computes institutional-grade value signals used in
equity research, factor models, and CIO dashboards.

Signals:
- Price-to-Earnings (P/E)
- Price-to-Book (P/B)
- Price-to-Sales (P/S)
- Dividend Yield
- Earnings Yield (inverse PE)
- Composite Value Score (normalized)
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


class ValueSignalEngine:
    """
    Production-ready value factor engine.

    Inputs:
    - Fundamental dataset (PE, PB, PS, dividend yield, EPS, BVPS, etc.)
    - Latest price data
    """

    REQUIRED_FIELDS = ["pe", "pb", "ps", "div_yield"]

    def compute_value_signals(
        self,
        symbol: str,
        as_of: pd.Timestamp,
        fundamentals: pd.Series,
        price: float,
    ) -> Dict[str, SignalPoint]:
        """
        Compute all value signals.

        Args:
            symbol: ticker
            as_of: timestamp
            fundamentals: pd.Series with required fields
            price: latest market price

        Returns:
            dict: signal_name -> SignalPoint
        """
        if fundamentals is None or fundamentals.empty:
            logger.warning("Value engine received empty fundamentals for %s", symbol)
            return {}

        signals: Dict[str, SignalPoint] = {}

        # Validate fields
        for f in self.REQUIRED_FIELDS:
            if f not in fundamentals:
                logger.warning("Missing fundamental field '%s' for %s", f, symbol)
                return {}

        try:
            # ---- PE ratio ----
            pe = fundamentals.get("pe", np.nan)
            if pe and pe > 0:
                signals["VALUE_PE"] = self._make_signal(
                    symbol, as_of, "VALUE_PE", float(pe), inverse=True
                )

            # ---- PB ratio ----
            pb = fundamentals.get("pb", np.nan)
            if pb and pb > 0:
                signals["VALUE_PB"] = self._make_signal(
                    symbol, as_of, "VALUE_PB", float(pb), inverse=True
                )

            # ---- PS ratio ----
            ps = fundamentals.get("ps", np.nan)
            if ps and ps > 0:
                signals["VALUE_PS"] = self._make_signal(
                    symbol, as_of, "VALUE_PS", float(ps), inverse=True
                )

            # ---- Dividend Yield ----
            dy = fundamentals.get("div_yield", np.nan)
            if dy and dy >= 0:
                signals["VALUE_DIVYLD"] = self._make_signal(
                    symbol, as_of, "VALUE_DIVYLD", float(dy), inverse=False
                )

            # ---- Composite Score ----
            comp = self._compute_composite(signals)
            if comp is not None:
                signals["VALUE_COMPOSITE"] = comp

        except Exception as e:
            logger.exception("Failed to compute value signals for %s: %s", symbol, e)

        return signals

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _make_signal(
        self,
        symbol: str,
        as_of: pd.Timestamp,
        name: str,
        value: float,
        inverse: bool = False,
    ) -> SignalPoint:
        """
        Create SignalPoint for a value metric.

        If inverse=True (PE, PB, PS): lower value = more attractive.
        """
        if inverse:
            norm = -value  # lower PE => higher normalized attractiveness
        else:
            norm = value

        # Determine direction
        if norm > 0:
            direction = SignalDirection.LONG
            strength = SignalStrength.POSITIVE
        elif norm < 0:
            direction = SignalDirection.SHORT
            strength = SignalStrength.NEGATIVE
        else:
            direction = SignalDirection.NEUTRAL
            strength = SignalStrength.NEUTRAL

        return SignalPoint(
            symbol=symbol,
            as_of=as_of,
            category=SignalCategory.VALUE,
            name=name,
            value=value,
            normalized_value=norm,
            direction=direction,
            strength=strength,
        )

    def _compute_composite(
        self,
        signals: Dict[str, SignalPoint],
    ) -> Optional[SignalPoint]:
        """
        Combine PE, PB, PS, DIVYLD into a single composite score.

        Approach:
        - Use normalized_value of each component
        - Average them
        """
        if not signals:
            return None

        norms = []
        for key in ["VALUE_PE", "VALUE_PB", "VALUE_PS", "VALUE_DIVYLD"]:
            sp = signals.get(key)
            if sp and sp.normalized_value is not None:
                norms.append(sp.normalized_value)

        if not norms:
            return None

        composite = float(np.mean(norms))
        direction = (
            SignalDirection.LONG
            if composite > 0
            else SignalDirection.SHORT
        )
        strength = (
            SignalStrength.POSITIVE
            if composite > 0
            else SignalStrength.NEGATIVE
        )

        # Create synthetic composite signal
        any_signal = next(iter(signals.values()))
        return SignalPoint(
            symbol=any_signal.symbol,
            as_of=any_signal.as_of,
            category=SignalCategory.VALUE,
            name="VALUE_COMPOSITE",
            value=composite,
            normalized_value=composite,
            direction=direction,
            strength=strength,
        )
