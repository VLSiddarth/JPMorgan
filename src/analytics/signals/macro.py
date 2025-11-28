# src/analytics/signals/macro.py

"""
Macro Signal Engine
-------------------

Computes macro / regime signals for regions like:
- Europe (EU)
- United States (US)
- Other regions/blocks

Inputs:
- Time series of macro indicators (GDP, inflation, unemployment, policy rate, yield curve)
- Configurable lookback window for normalization

Outputs:
- MacroSignal objects (for CIO dashboards & top-down allocation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.data.models.signals import MacroSignal

logger = logging.getLogger(__name__)


@dataclass
class MacroInputs:
    """
    Container for macro time series for a region.

    All series are expected as pandas.Series indexed by datetime or date.
    Values should be numeric (floats), with missing values handled upstream
    as far as possible.
    """

    gdp_growth: Optional[pd.Series] = None          # Real GDP growth (QoQ or YoY)
    inflation: Optional[pd.Series] = None           # Headline CPI YoY
    core_inflation: Optional[pd.Series] = None      # Core CPI/HICP YoY (optional)
    unemployment: Optional[pd.Series] = None        # Unemployment rate (%)
    policy_rate: Optional[pd.Series] = None         # Central bank policy rate
    yield_curve_10y_2y: Optional[pd.Series] = None  # 10y - 2y spread
    yield_curve_10y_3m: Optional[pd.Series] = None  # 10y - 3m spread (optional)


class MacroSignalEngine:
    """
    Production-grade macro signal engine.

    design:
        - Accepts time series of key indicators
        - Normalizes them over a configurable lookback window
        - Produces a MacroSignal with scores and a composite risk-on score
    """

    def __init__(self, lookback_months: int = 60) -> None:
        """
        Args:
            lookback_months: rolling window to compute z-scores (approx)
        """
        self.lookback_months = lookback_months

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def compute_region_macro_signal(
        self,
        region: str,
        inputs: MacroInputs,
        as_of: Optional[date] = None,
    ) -> Optional[MacroSignal]:
        """
        Compute MacroSignal for a given region.

        Args:
            region: region code (e.g., 'EU', 'US')
            inputs: MacroInputs with relevant time series
            as_of: date to evaluate regime (default: latest common date)

        Returns:
            MacroSignal or None if insufficient data
        """
        try:
            # Pick reference date
            ref_date = self._determine_as_of(inputs, as_of)
            if ref_date is None:
                logger.warning("MacroSignalEngine: no valid as_of date for region=%s", region)
                return None

            # Compute scores
            growth_score = self._compute_growth_score(inputs.gdp_growth, inputs.unemployment, ref_date)
            inflation_score = self._compute_inflation_score(inputs.inflation, inputs.core_inflation, ref_date)
            policy_score = self._compute_policy_score(inputs.policy_rate, ref_date)
            curve_score = self._compute_curve_score(inputs.yield_curve_10y_2y, inputs.yield_curve_10y_3m, ref_date)

            if any(np.isnan([growth_score, inflation_score, policy_score, curve_score])):
                logger.warning("MacroSignalEngine: insufficient data to compute full macro signal for %s", region)

            risk_on = self._combine_scores(
                growth_score=growth_score,
                inflation_score=inflation_score,
                policy_score=policy_score,
                curve_score=curve_score,
            )

            narrative = self._build_narrative(
                region=region,
                ref_date=ref_date,
                growth_score=growth_score,
                inflation_score=inflation_score,
                policy_score=policy_score,
                curve_score=curve_score,
                risk_on=risk_on,
            )

            return MacroSignal(
                region=region,
                as_of=ref_date,
                growth_score=float(growth_score),
                inflation_score=float(inflation_score),
                policy_score=float(policy_score),
                curve_score=float(curve_score),
                risk_on_score=float(risk_on),
                narrative=narrative,
                metadata={
                    "lookback_months": self.lookback_months,
                },
            )
        except Exception as exc:
            logger.exception("Failed to compute MacroSignal for region=%s: %s", region, exc)
            return None

    # ----------------------------------------------------------------------
    # Internal scoring helpers
    # ----------------------------------------------------------------------
    def _determine_as_of(self, inputs: MacroInputs, as_of: Optional[date]) -> Optional[date]:
        """Determine the reference date for the macro regime."""
        if as_of is not None:
            return as_of

        candidates = []
        for series in [
            inputs.gdp_growth,
            inputs.inflation,
            inputs.core_inflation,
            inputs.unemployment,
            inputs.policy_rate,
            inputs.yield_curve_10y_2y,
            inputs.yield_curve_10y_3m,
        ]:
            if series is not None and not series.empty:
                candidates.append(series.dropna().index.max())

        if not candidates:
            return None

        ref_ts = max(candidates)
        return ref_ts.date() if hasattr(ref_ts, "date") else ref_ts

    def _zscore(
        self,
        series: Optional[pd.Series],
        ref_date: date,
        window_months: int,
        clip: float = 3.0,
    ) -> float:
        """
        Compute z-score of the latest point using historical window.

        series: monthly or quarterly data; uses last window_months worth of points.
        """
        if series is None or series.empty:
            return np.nan

        s = series.dropna()
        if s.empty:
            return np.nan

        # Filter for data up to reference date
        s = s.loc[s.index <= pd.Timestamp(ref_date)]

        if s.empty:
            return np.nan

        # Use last N observations (approx lookback_months)
        window_size = min(len(s), max(12, window_months // 1))
        tail = s.iloc[-window_size:]

        latest = tail.iloc[-1]
        mu = tail.mean()
        sigma = tail.std()

        if sigma == 0 or np.isnan(sigma):
            return 0.0

        z = (latest - mu) / sigma
        # Clip extreme values
        z = float(np.clip(z, -clip, clip))
        return z

    def _compute_growth_score(
        self,
        gdp_growth: Optional[pd.Series],
        unemployment: Optional[pd.Series],
        ref_date: date,
    ) -> float:
        """
        Growth score: combines GDP growth and change in unemployment.

        Intuition:
        - Higher GDP growth => positive score
        - Falling unemployment => positive score
        """
        z_gdp = self._zscore(gdp_growth, ref_date, self.lookback_months) if gdp_growth is not None else np.nan

        z_unemp = np.nan
        if unemployment is not None and not unemployment.empty:
            # Use change in unemployment (inverted)
            s = unemployment.dropna()
            s = s.loc[s.index <= pd.Timestamp(ref_date)]
            if len(s) >= 6:
                delta = s.diff().iloc[-3:]  # last few changes
                # negative changes (falling unemployment) are good
                # so take negative of z-score
                z_unemp = -self._zscore(delta, ref_date, window_months=12)

        if np.isnan(z_gdp) and np.isnan(z_unemp):
            return np.nan
        if np.isnan(z_unemp):
            return z_gdp
        if np.isnan(z_gdp):
            return z_unemp

        return float(0.6 * z_gdp + 0.4 * z_unemp)

    def _compute_inflation_score(
        self,
        inflation: Optional[pd.Series],
        core_inflation: Optional[pd.Series],
        ref_date: date,
    ) -> float:
        """
        Inflation score: captures inflation pressure.

        Convention:
        - Higher inflation => more negative score (risk-off)
        """
        z_headline = self._zscore(inflation, ref_date, self.lookback_months)
        z_core = self._zscore(core_inflation, ref_date, self.lookback_months) if core_inflation is not None else np.nan

        if np.isnan(z_headline) and np.isnan(z_core):
            return np.nan
        if np.isnan(z_core):
            z_eff = z_headline
        elif np.isnan(z_headline):
            z_eff = z_core
        else:
            z_eff = 0.7 * z_headline + 0.3 * z_core

        # Higher inflation => worse for risk (negative contribution)
        return float(-z_eff)

    def _compute_policy_score(
        self,
        policy_rate: Optional[pd.Series],
        ref_date: date,
    ) -> float:
        """
        Policy score: captures stance and change in policy.

        Intuition:
        - Higher/lower than historical mean matters less than direction
        - Recent hikes => negative, cuts => positive
        """
        if policy_rate is None or policy_rate.empty:
            return np.nan

        s = policy_rate.dropna()
        s = s.loc[s.index <= pd.Timestamp(ref_date)]
        if len(s) < 6:
            return np.nan

        z_level = self._zscore(s, ref_date, self.lookback_months)
        # Change in last 3 obs
        delta = s.diff().iloc[-3:]
        z_delta = self._zscore(delta, ref_date, window_months=12)

        # Higher level + recent hikes => restrictive (negative)
        score = -0.5 * z_level - 0.5 * z_delta
        return float(score)

    def _compute_curve_score(
        self,
        curve_10y_2y: Optional[pd.Series],
        curve_10y_3m: Optional[pd.Series],
        ref_date: date,
    ) -> float:
        """
        Curve score: steep vs inverted curves.

        Intuition:
        - Positive, steep curve => risk-on (+)
        - Deep inversion => risk-off (-)
        """
        z_10_2 = self._zscore(curve_10y_2y, ref_date, self.lookback_months)
        z_10_3 = self._zscore(curve_10y_3m, ref_date, self.lookback_months) if curve_10y_3m is not None else np.nan

        if np.isnan(z_10_2) and np.isnan(z_10_3):
            return np.nan
        if np.isnan(z_10_3):
            z_eff = z_10_2
        elif np.isnan(z_10_2):
            z_eff = z_10_3
        else:
            z_eff = 0.6 * z_10_2 + 0.4 * z_10_3

        # Steeper curve (higher spread) => more positive score
        return float(z_eff)

    def _combine_scores(
        self,
        growth_score: float,
        inflation_score: float,
        policy_score: float,
        curve_score: float,
    ) -> float:
        """
        Compute composite risk-on score from component scores.

        A simple weighted sum with equal-ish importance. This can be tuned.
        """
        components = np.array([growth_score, inflation_score, policy_score, curve_score], dtype=float)
        mask = ~np.isnan(components)
        if not mask.any():
            return np.nan

        weights = np.array([0.4, 0.2, 0.2, 0.2])  # growth more important
        weights = weights[mask]
        components = components[mask]

        # normalize weights to sum to 1
        weights = weights / weights.sum()

        score = float(np.dot(weights, components))
        # clip to [-3, 3] range for interpretability
        return float(np.clip(score, -3.0, 3.0))

    def _build_narrative(
        self,
        region: str,
        ref_date: date,
        growth_score: float,
        inflation_score: float,
        policy_score: float,
        curve_score: float,
        risk_on: float,
    ) -> str:
        """
        Build a short, human-readable macro narrative for dashboards.
        """
        def classify(x: float, positive_label: str, negative_label: str) -> str:
            if np.isnan(x):
                return "neutral"
            if x > 1.0:
                return f"strong_{positive_label}"
            if x > 0.3:
                return positive_label
            if x < -1.0:
                return f"strong_{negative_label}"
            if x < -0.3:
                return negative_label
            return "neutral"

        growth_state = classify(growth_score, "expansion", "slowdown")
        inflation_state = classify(inflation_score, "disinflation", "inflation_pressure")
        policy_state = classify(policy_score, "accommodative", "restrictive")
        curve_state = classify(curve_score, "steep_curve", "inverted_curve")
        risk_state = classify(risk_on, "risk_on", "risk_off")

        narrative = (
            f"{region} macro regime as of {ref_date}: "
            f"growth={growth_state}, "
            f"inflation={inflation_state}, "
            f"policy={policy_state}, "
            f"curve={curve_state}, "
            f"overall={risk_state}."
        )
        return narrative
