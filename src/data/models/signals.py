# src/data/models/signals.py

"""
Signal data models for quantitative strategies and CIO dashboards.

These models represent:
- Point-in-time signals for a single instrument
- Cross-sectional signal snapshots
- Time series of signals for backtesting and monitoring
"""

from __future__ import annotations

from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


class SignalCategory(str, Enum):
    """High-level signal type."""

    MOMENTUM = "momentum"
    VALUE = "value"
    QUALITY = "quality"
    SIZE = "size"
    VOLATILITY = "volatility"
    MACRO = "macro"
    SENTIMENT = "sentiment"
    RISK = "risk"
    OTHER = "other"


class SignalDirection(str, Enum):
    """Direction of signal."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    OVERWEIGHT = "overweight"
    UNDERWEIGHT = "underweight"


class SignalStrength(str, Enum):
    """Qualitative strength mapping for CIO dashboards."""

    STRONG_POSITIVE = "strong_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    STRONG_NEGATIVE = "strong_negative"


class SignalPoint(BaseModel):
    """
    Single signal observation for a symbol at a given timestamp.
    """

    symbol: str = Field(..., description="Ticker symbol")
    as_of: datetime = Field(..., description="Timestamp of signal computation")

    category: SignalCategory = Field(..., description="Signal category")
    name: str = Field(..., description="Signal name (e.g., 'MOM_12M', 'VALUE_PE')")
    value: float = Field(..., description="Raw numeric signal value")

    normalized_value: Optional[float] = Field(
        default=None,
        description="Cross-sectionally normalized value (e.g., z-score)",
    )
    direction: SignalDirection = Field(
        default=SignalDirection.NEUTRAL,
        description="Direction implied by the signal",
    )
    strength: SignalStrength = Field(
        default=SignalStrength.NEUTRAL,
        description="Qualitative strength for dashboards",
    )

    lookback_days: Optional[int] = Field(
        default=None,
        description="Lookback period used for signal calculation (if applicable)",
    )
    universe_id: Optional[str] = Field(
        default=None,
        description="Universe identifier (e.g., 'EUROPE_LARGE_CAP')",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional fields (ranks, deciles, raw stats, etc.)",
    )

    @validator("name")
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Signal name cannot be empty")
        return v.strip()

    class Config:
        validate_assignment = True


class SymbolSignalHistory(BaseModel):
    """
    Time series of a given signal for a single symbol.
    """

    symbol: str = Field(..., description="Ticker symbol")
    signal_name: str = Field(..., description="Signal name")
    category: SignalCategory = Field(..., description="Signal category")
    points: List[SignalPoint] = Field(
        default_factory=list,
        description="Chronologically ordered signal points",
    )

    @validator("points")
    def validate_sorted(cls, v: List[SignalPoint]) -> List[SignalPoint]:
        if not v:
            return v
        sorted_pts = sorted(v, key=lambda p: p.as_of)
        return sorted_pts

    class Config:
        validate_assignment = True


class CrossSectionalSignalSnapshot(BaseModel):
    """
    Cross-sectional snapshot of a signal across a universe.

    Used for ranking, portfolio construction, heatmaps, etc.
    """

    as_of: datetime = Field(..., description="Timestamp of snapshot")
    signal_name: str = Field(..., description="Signal name")
    category: SignalCategory = Field(..., description="Signal category")
    universe_id: Optional[str] = Field(
        default=None, description="Universe ID (e.g., 'EUROPE_LARGE_CAP')"
    )

    signals: List[SignalPoint] = Field(
        default_factory=list,
        description="Signals for each symbol at this timestamp",
    )

    def to_rank_dict(self, descending: bool = True) -> Dict[str, int]:
        """
        Compute ranks for each symbol based on normalized_value or value.

        Returns:
            dict: symbol -> rank (1 = best)
        """
        import pandas as pd

        if not self.signals:
            return {}

        records = []
        for sp in self.signals:
            records.append(
                {
                    "symbol": sp.symbol,
                    "value": sp.normalized_value
                    if sp.normalized_value is not None
                    else sp.value,
                }
            )

        df = pd.DataFrame.from_records(records)
        df = df.dropna(subset=["value"])

        if df.empty:
            return {}

        df["rank"] = df["value"].rank(ascending=not descending, method="min")
        return {row.symbol: int(row.rank) for row in df.itertuples(index=False)}

    class Config:
        validate_assignment = True


class MacroSignal(BaseModel):
    """
    Macro / regime signal at a region or country level.

    This captures macro context used for top-down allocation decisions.
    """

    region: str = Field(..., description="Region/country code, e.g. 'EU', 'US'")
    as_of: date = Field(..., description="Date of the macro reading")
    growth_score: float = Field(
        ..., description="Normalized growth score (-3 to +3, for example)"
    )
    inflation_score: float = Field(
        ..., description="Normalized inflation score (-3 to +3)"
    )
    policy_score: float = Field(
        ..., description="Normalized policy stance score (-3 to +3)"
    )
    curve_score: float = Field(
        ..., description="Yield curve slope / steepness score (-3 to +3)"
    )

    risk_on_score: float = Field(
        ..., description="Composite risk-on score derived from above"
    )
    narrative: Optional[str] = Field(
        default=None,
        description="Short textual description of macro regime",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional macro indicators used",
    )

    class Config:
        validate_assignment = True
