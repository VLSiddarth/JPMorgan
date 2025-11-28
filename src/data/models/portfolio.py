"""
src/data/models/portfolio.py

Pydantic models for portfolio and position data, suitable for a JPMorgan-grade
European equity dashboard.

These models are used across:
- Portfolio construction & optimization
- Risk analytics (exposure, concentration)
- Backtesting & performance attribution
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator, root_validator
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Enums & shared types
# ──────────────────────────────────────────────────────────────────────────────


class PortfolioType(str, Enum):
    """High-level portfolio type."""

    MODEL = "model"
    LIVE = "live"
    BENCHMARK = "benchmark"
    STRATEGY = "strategy"


class PositionSide(str, Enum):
    """Direction of the position."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class Currency(str, Enum):
    """Common currencies; can be extended."""

    EUR = "EUR"
    USD = "USD"
    GBP = "GBP"
    CHF = "CHF"
    JPY = "JPY"
    OTHER = "OTHER"


# ──────────────────────────────────────────────────────────────────────────────
# Positions & holdings
# ──────────────────────────────────────────────────────────────────────────────


class Position(BaseModel):
    """
    Single instrument position.

    This is a "unit-based" representation: quantity, price, currency. It can be
    converted to value using mark-to-market prices.
    """

    symbol: str = Field(..., description="Ticker, e.g. 'SX5E' or '^STOXX50E'")
    side: PositionSide = Field(
        default=PositionSide.LONG, description="Position direction"
    )
    quantity: float = Field(
        default=0.0,
        description="Number of units (shares, contracts, etc.)",
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Average entry price in local currency",
    )
    currency: Currency | str = Field(
        default=Currency.EUR,
        description="Trading currency of position",
    )
    sector: Optional[str] = Field(
        default=None,
        description="GICS sector or similar, for aggregation",
    )
    country: Optional[str] = Field(
        default=None,
        description="Country ISO code (e.g. DE, FR)",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Arbitrary labels (e.g., 'GRANOLAS', 'banks')",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra, non-standard position metadata",
    )

    @validator("quantity")
    def validate_quantity(cls, v: float, values: Dict[str, Any]) -> float:
        side = values.get("side", PositionSide.LONG)
        if v < 0:
            raise ValueError("Quantity cannot be negative; use side=SHORT instead")
        if side == PositionSide.FLAT and v != 0:
            raise ValueError("FLAT position must have quantity=0")
        return v

    @property
    def is_flat(self) -> bool:
        """Return True if the position has effectively zero exposure."""
        return self.quantity == 0 or self.side == PositionSide.FLAT

    class Config:
        validate_assignment = True


class PositionValuation(BaseModel):
    """
    Valuation of a position at a given point in time.

    This is "mark-to-market" representation, derived from Position + market data.
    """

    symbol: str = Field(..., description="Ticker")
    as_of: datetime = Field(..., description="Valuation timestamp")
    currency: Currency | str = Field(
        default=Currency.EUR, description="Valuation currency"
    )

    # Inputs
    quantity: float = Field(..., description="Units held (same as Position.quantity)")
    side: PositionSide = Field(..., description="Position direction")
    price: float = Field(..., description="Mark price in currency")
    fx_rate_to_portfolio_ccy: float = Field(
        default=1.0,
        description="FX rate from position currency to portfolio base currency",
    )

    # Derived metrics
    market_value: float = Field(
        ..., description="Mark-to-market value in position currency"
    )
    market_value_base: float = Field(
        ..., description="Mark-to-market value in portfolio base currency"
    )
    weight: Optional[float] = Field(
        default=None,
        description="Weight in portfolio (% of NAV), if known",
    )

    @classmethod
    def from_position(
        cls,
        position: Position,
        as_of: datetime,
        price: float,
        portfolio_base_ccy: Currency | str = Currency.EUR,
        fx_rate: float = 1.0,
    ) -> "PositionValuation":
        """
        Create a PositionValuation from a Position and a price/FX rate.

        Args:
            position: Position object
            as_of: valuation timestamp
            price: mark price in position currency
            portfolio_base_ccy: base currency of the portfolio
            fx_rate: FX rate from position currency to portfolio base currency

        Returns:
            PositionValuation instance
        """
        qty = position.quantity
        if position.side == PositionSide.SHORT:
            qty = -qty

        mv_local = qty * price
        mv_base = mv_local * fx_rate

        return cls(
            symbol=position.symbol,
            as_of=as_of,
            currency=portfolio_base_ccy,
            quantity=position.quantity,
            side=position.side,
            price=price,
            fx_rate_to_portfolio_ccy=fx_rate,
            market_value=mv_local,
            market_value_base=mv_base,
        )

    class Config:
        validate_assignment = True


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio models
# ──────────────────────────────────────────────────────────────────────────────


class PortfolioConstraints(BaseModel):
    """
    Portfolio-level constraints used by optimizers and risk engines.
    """

    max_weight: float = Field(
        default=0.10, ge=0.0, le=1.0, description="Max single-name weight"
    )
    min_weight: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Min single-name weight"
    )
    max_gross_exposure: float = Field(
        default=2.0, ge=0.0, description="Max gross exposure (e.g., 2.0 = 200%)"
    )
    max_net_exposure: float = Field(
        default=1.0, ge=-1.0, le=2.0, description="Max net exposure"
    )
    max_sector_weight: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Max per-sector weight"
    )
    max_position_count: Optional[int] = Field(
        default=None, description="Max number of positions allowed"
    )

    class Config:
        validate_assignment = True


class PortfolioHolding(BaseModel):
    """
    A stable "weight-based" representation used for optimization and backtests.
    """

    symbol: str = Field(..., description="Ticker")
    weight: float = Field(..., description="Portfolio weight (can be negative)")
    currency: Currency | str = Field(
        default=Currency.EUR, description="Currency of the instrument"
    )
    sector: Optional[str] = Field(default=None, description="Sector label")
    country: Optional[str] = Field(default=None, description="Country code")
    benchmark_weight: Optional[float] = Field(
        default=None, description="Corresponding benchmark weight, if any"
    )

    @validator("weight")
    def validate_weight(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            logger.warning("Weight %.4f outside [-1, 1] range", v)
        return v

    class Config:
        validate_assignment = True


class PortfolioSnapshot(BaseModel):
    """
    Snapshot of the portfolio at a point in time.

    Holds positions in both "unit" and "weight" terms, plus NAV.
    """

    portfolio_id: str = Field(..., description="Unique portfolio identifier")
    as_of: datetime = Field(..., description="Timestamp of the snapshot")
    base_currency: Currency | str = Field(
        default=Currency.EUR, description="Base currency of portfolio"
    )
    portfolio_type: PortfolioType = Field(
        default=PortfolioType.MODEL,
        description="Type of the portfolio (model/live/benchmark/strategy)",
    )

    nav: float = Field(..., gt=0.0, description="Net Asset Value in base currency")

    holdings: List[PortfolioHolding] = Field(
        default_factory=list,
        description="Holdings with weights in portfolio",
    )
    # Optional original positions (if needed for unit-level details)
    positions: Optional[List[Position]] = Field(
        default=None,
        description="Optional list of unit-based positions",
    )

    constraints: Optional[PortfolioConstraints] = Field(
        default=None,
        description="Constraints applicable to this portfolio snapshot",
    )

    tags: List[str] = Field(
        default_factory=list,
        description="Arbitrary labels (e.g., 'European_equity', 'GRANOLAS_focus')",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (strategy name, mandate, etc.)",
    )

    @root_validator
    def validate_holdings_sum(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        holdings: List[PortfolioHolding] = values.get("holdings", [])
        if holdings:
            total_weight = sum(h.weight for h in holdings)
            if not 0.95 <= total_weight <= 1.05:
                logger.warning(
                    "Portfolio weights sum to %.4f (outside [0.95, 1.05])",
                    total_weight,
                )
        return values

    def get_weight_dict(self) -> Dict[str, float]:
        """Return mapping symbol -> weight."""
        return {h.symbol: h.weight for h in self.holdings}

    def to_positions_valuation(
        self,
        prices: Dict[str, float],
        fx_rates: Optional[Dict[str, float]] = None,
    ) -> List[PositionValuation]:
        """
        Convert holdings into PositionValuation objects given current prices and FX.

        Args:
            prices: mapping symbol -> price in instrument currency
            fx_rates: mapping symbol -> FX rate to base currency (if None, assume 1.0)

        Returns:
            List[PositionValuation]
        """
        fx_rates = fx_rates or {}
        valuations: List[PositionValuation] = []

        for h in self.holdings:
            price = prices.get(h.symbol)
            if price is None:
                logger.warning(
                    "Missing price for symbol %s in portfolio %s",
                    h.symbol,
                    self.portfolio_id,
                )
                continue

            fx_rate = fx_rates.get(h.symbol, 1.0)
            # Convert weight to notional: weight * NAV
            notional_base = h.weight * self.nav
            qty = notional_base / (price * fx_rate) if price * fx_rate != 0 else 0.0

            pos = Position(
                symbol=h.symbol,
                side=PositionSide.LONG if h.weight >= 0 else PositionSide.SHORT,
                quantity=abs(qty),
                currency=h.currency,
                sector=h.sector,
                country=h.country,
            )

            valuation = PositionValuation.from_position(
                position=pos,
                as_of=self.as_of,
                price=price,
                portfolio_base_ccy=self.base_currency,
                fx_rate=fx_rate,
            )
            valuations.append(valuation)

        return valuations

    class Config:
        validate_assignment = True


class PortfolioPerformancePoint(BaseModel):
    """
    Time series point for portfolio performance.

    This is used in backtests and live performance reporting.
    """

    portfolio_id: str = Field(..., description="Portfolio identifier")
    date: date = Field(..., description="Calendar date")
    nav: float = Field(..., gt=0.0, description="Net Asset Value at end of day")
    return_daily: float = Field(..., description="Daily return")
    benchmark_return_daily: Optional[float] = Field(
        default=None, description="Benchmark daily return, if any"
    )
    gross_exposure: Optional[float] = Field(
        default=None, description="Gross exposure (e.g. 1.2 = 120%)"
    )
    net_exposure: Optional[float] = Field(
        default=None, description="Net exposure (e.g. 0.7 = 70%)"
    )

    class Config:
        validate_assignment = True


class PortfolioPerformanceSeries(BaseModel):
    """
    Collection of PortfolioPerformancePoint, e.g. for a backtest or live history.
    """

    portfolio_id: str = Field(..., description="Portfolio identifier")
    points: List[PortfolioPerformancePoint] = Field(
        default_factory=list,
        description="Chronologically ordered performance points",
    )

    @validator("points")
    def validate_sorted_unique_dates(
        cls, v: List[PortfolioPerformancePoint]
    ) -> List[PortfolioPerformancePoint]:
        if not v:
            return v
        sorted_pts = sorted(v, key=lambda p: p.date)
        dates = [p.date for p in sorted_pts]
        if len(dates) != len(set(dates)):
            raise ValueError("Duplicate dates in performance series")
        return sorted_pts

    def to_dataframe(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        """
        Convert the performance series to a pandas DataFrame.

        Returns:
            DataFrame indexed by date with columns (nav, return_daily, etc.)
        """
        import pandas as pd

        if not self.points:
            return pd.DataFrame(
                columns=[
                    "nav",
                    "return_daily",
                    "benchmark_return_daily",
                    "gross_exposure",
                    "net_exposure",
                ]
            )

        records = [asdict(p) for p in self.points]  # dataclasses-style for BaseModel
        # asdict doesn't work directly on BaseModel; convert manually:
        records = [p.dict() for p in self.points]
        df = pd.DataFrame.from_records(records).set_index("date").sort_index()
        return df

    class Config:
        validate_assignment = True
