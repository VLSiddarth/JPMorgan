"""
src/data/models/market_data.py

Pydantic models for market data, suitable for institutional / JPMorgan-grade
systems. These models are used across connectors, repositories, and analytics
layers to enforce a consistent, validated schema for market data.

Key concepts:
- Instrument metadata (what is this ticker?)
- Price bars (OHLCV)
- Time series / collections of bars
- Market data requests / responses
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Enums & constants
# ──────────────────────────────────────────────────────────────────────────────


class MarketDataProvider(str, Enum):
    """Supported market data providers (free by default)."""

    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    FRED = "fred"
    ECB_SDW = "ecb_sdw"
    CUSTOM = "custom"


class AssetClass(str, Enum):
    """High-level asset classification."""

    EQUITY = "equity"
    INDEX = "index"
    ETF = "etf"
    FX = "fx"
    BOND = "bond"
    COMMODITY = "commodity"
    OTHER = "other"


class PriceField(str, Enum):
    """Standard price fields."""

    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    ADJ_CLOSE = "adj_close"
    VOLUME = "volume"


class BarInterval(str, Enum):
    """Supported bar intervals."""

    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1wk"
    MONTH_1 = "1mo"


# ──────────────────────────────────────────────────────────────────────────────
# Instrument metadata
# ──────────────────────────────────────────────────────────────────────────────


class InstrumentMetadata(BaseModel):
    """
    Static metadata for a traded instrument or index.

    This is typically fetched from the data provider once and cached, and used
    across analytics for sector/region attribution, currency handling, etc.
    """

    symbol: str = Field(..., description="Native ticker symbol (e.g. ^STOXX50E)")
    name: Optional[str] = Field(
        default=None, description="Human-readable name (e.g. EURO STOXX 50)"
    )
    asset_class: AssetClass = Field(
        default=AssetClass.INDEX, description="High-level asset class"
    )
    currency: Optional[str] = Field(
        default=None, description="Trading currency (e.g. EUR, USD)"
    )
    exchange: Optional[str] = Field(
        default=None, description="Primary exchange or venue (e.g. XETRA)"
    )
    country: Optional[str] = Field(
        default=None, description="Country ISO code (e.g. DE, FR, EU)"
    )
    sector: Optional[str] = Field(
        default=None, description="GICS sector or equivalent (for equities/ETFs)"
    )
    industry: Optional[str] = Field(
        default=None, description="Industry or sub-sector (for equities/ETFs)"
    )
    provider: MarketDataProvider = Field(
        default=MarketDataProvider.YAHOO,
        description="Source of this metadata",
    )
    is_primary: bool = Field(
        default=True,
        description="True if this is the primary listing / representation",
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific optional fields",
    )

    class Config:
        frozen = True  # hashable, safe for caching
        validate_assignment = True


# ──────────────────────────────────────────────────────────────────────────────
# Price bar & series models
# ──────────────────────────────────────────────────────────────────────────────


class PriceBar(BaseModel):
    """
    Single OHLCV bar for one instrument at a given timestamp.

    All price fields are expressed in the instrument's currency.
    """

    timestamp: datetime = Field(..., description="Bar timestamp (UTC recommended)")

    open: Optional[float] = Field(
        default=None,
        description="Open price",
    )
    high: Optional[float] = Field(
        default=None,
        description="High price",
    )
    low: Optional[float] = Field(
        default=None,
        description="Low price",
    )
    close: Optional[float] = Field(
        default=None,
        description="Close price",
    )
    adj_close: Optional[float] = Field(
        default=None,
        description="Adjusted close price (if available)",
    )
    volume: Optional[float] = Field(
        default=None,
        ge=0,
        description="Traded volume (units, not currency)",
    )

    provider: MarketDataProvider = Field(
        default=MarketDataProvider.YAHOO,
        description="Data source for this bar",
    )

    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific extra fields (bid/ask, open interest, etc.)",
    )

    @validator("timestamp")
    def validate_timestamp(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            # For internal consistency, allow naive but log a warning.
            logger.debug("PriceBar timestamp is naive (no timezone). Consider using UTC.")
        return v

    @validator("high")
    def validate_high(cls, v: Optional[float], values: Dict[str, Any]) -> Optional[float]:
        low = values.get("low")
        if v is not None and low is not None and v < low:
            raise ValueError("High price cannot be lower than low price")
        return v

    @validator("low")
    def validate_low(cls, v: Optional[float], values: Dict[str, Any]) -> Optional[float]:
        high = values.get("high")
        if v is not None and high is not None and v > high:
            raise ValueError("Low price cannot be higher than high price")
        return v

    class Config:
        validate_assignment = True
        allow_mutation = True
        arbitrary_types_allowed = True


class HistoricalPriceSeries(BaseModel):
    """
    Ordered collection of price bars for a single instrument.

    This object is the canonical representation of a time series of prices in
    the application layer. It provides helpers to convert to/from DataFrames.
    """

    symbol: str = Field(..., description="Ticker symbol")
    metadata: Optional[InstrumentMetadata] = Field(
        default=None,
        description="Optional instrument metadata",
    )
    interval: BarInterval = Field(
        default=BarInterval.DAY_1,
        description="Bar interval for the series",
    )
    bars: List[PriceBar] = Field(
        default_factory=list,
        description="Time-ordered list of price bars",
    )

    @validator("bars")
    def validate_sorted_unique(cls, v: List[PriceBar]) -> List[PriceBar]:
        """Ensure bars are sorted by timestamp and unique."""
        if not v:
            return v

        sorted_bars = sorted(v, key=lambda b: b.timestamp)
        timestamps = [b.timestamp for b in sorted_bars]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError("Duplicate timestamps in price series")

        return sorted_bars

    @property
    def start(self) -> Optional[datetime]:
        """First bar timestamp, if any."""
        return self.bars[0].timestamp if self.bars else None

    @property
    def end(self) -> Optional[datetime]:
        """Last bar timestamp, if any."""
        return self.bars[-1].timestamp if self.bars else None

    def to_dataframe(self) -> "pd.DataFrame":  # type: ignore[name-defined]
        """
        Convert the price series to a pandas DataFrame.

        Returns:
            DataFrame indexed by timestamp with columns:
            ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        """
        import pandas as pd  # lazy import to keep model lightweight

        if not self.bars:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "adj_close", "volume"]
            )

        records = [
            {
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "adj_close": b.adj_close,
                "volume": b.volume,
            }
            for b in self.bars
        ]
        df = pd.DataFrame.from_records(records).set_index("timestamp").sort_index()
        return df

    @classmethod
    def from_dataframe(
        cls,
        symbol: str,
        df: "pd.DataFrame",  # type: ignore[name-defined]
        interval: BarInterval = BarInterval.DAY_1,
        metadata: Optional[InstrumentMetadata] = None,
        provider: MarketDataProvider = MarketDataProvider.YAHOO,
    ) -> "HistoricalPriceSeries":
        """
        Build a HistoricalPriceSeries from a DataFrame.

        Expected columns (case-insensitive, subset allowed):
            ['open', 'high', 'low', 'close', 'adj_close', 'volume']

        Args:
            symbol: Ticker symbol
            df: DataFrame indexed by datetime
            interval: Bar interval
            metadata: Optional InstrumentMetadata
            provider: MarketDataProvider

        Returns:
            HistoricalPriceSeries
        """
        import pandas as pd

        if df is None or df.empty:
            logger.warning("from_dataframe received empty DataFrame for symbol=%s", symbol)
            return cls(symbol=symbol, metadata=metadata, interval=interval, bars=[])

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex")

        normalized_cols = {c.lower(): c for c in df.columns}
        bars: List[PriceBar] = []

        for ts, row in df.sort_index().iterrows():
            bar_kwargs: Dict[str, Any] = {
                "timestamp": ts,
                "provider": provider,
            }
            for field in ["open", "high", "low", "close", "adj_close", "volume"]:
                col = normalized_cols.get(field)
                if col is not None:
                    value = row[col]
                    bar_kwargs[field] = None if pd.isna(value) else float(value)

            try:
                bar = PriceBar(**bar_kwargs)
                bars.append(bar)
            except Exception as exc:
                logger.error(
                    "Failed to construct PriceBar for %s at %s: %s", symbol, ts, exc
                )
                # optionally skip invalid bar instead of failing entire series
                continue

        return cls(symbol=symbol, metadata=metadata, interval=interval, bars=bars)

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True


# ──────────────────────────────────────────────────────────────────────────────
# Request / response models for data layer
# ──────────────────────────────────────────────────────────────────────────────


class MarketDataRequest(BaseModel):
    """
    Request model for fetching market data from a connector or repository.
    """

    symbols: List[str] = Field(..., description="List of ticker symbols")
    start_date: Optional[date] = Field(
        default=None, description="Start date (inclusive) in local calendar"
    )
    end_date: Optional[date] = Field(
        default=None, description="End date (inclusive) in local calendar"
    )
    interval: BarInterval = Field(
        default=BarInterval.DAY_1, description="Requested bar interval"
    )
    fields: Optional[List[PriceField]] = Field(
        default=None,
        description=(
            "Subset of price fields. If None, connector should return all standard "
            "fields it supports."
        ),
    )
    provider: Optional[MarketDataProvider] = Field(
        default=None,
        description="Optional explicit provider override; if None, use default",
    )
    adjusted: bool = Field(
        default=True,
        description="Whether to request adjusted prices (if supported by provider)",
    )

    @validator("symbols")
    def validate_symbols(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one symbol must be provided")
        cleaned = [s.strip() for s in v if s and s.strip()]
        if not cleaned:
            raise ValueError("Symbols cannot be empty or whitespace")
        return cleaned

    @validator("end_date")
    def validate_date_range(
        cls, v: Optional[date], values: Dict[str, Any]
    ) -> Optional[date]:
        start = values.get("start_date")
        if v is not None and start is not None and v < start:
            raise ValueError("end_date cannot be earlier than start_date")
        return v

    class Config:
        validate_assignment = True


class MarketDataError(BaseModel):
    """
    Error object for partial failures in multi-symbol responses.
    """

    symbol: str = Field(..., description="Ticker symbol for which error occurred")
    code: str = Field(..., description="Short error code (e.g., NOT_FOUND, TIMEOUT)")
    message: str = Field(..., description="Human-readable error description")
    provider: Optional[MarketDataProvider] = Field(
        default=None, description="Provider where the error occurred"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional provider-specific or diagnostic information",
    )


class MarketDataResponse(BaseModel):
    """
    Response model for market data fetches, supporting partial success.

    A single request might succeed for some symbols and fail for others; both
    are captured here.
    """

    request: MarketDataRequest = Field(
        ..., description="Original request parameters"
    )
    series: Dict[str, HistoricalPriceSeries] = Field(
        default_factory=dict,
        description="Mapping from symbol to HistoricalPriceSeries",
    )
    errors: List[MarketDataError] = Field(
        default_factory=list,
        description="List of errors for symbols that could not be fetched",
    )

    def is_success(self) -> bool:
        """Return True if all requested symbols were successfully fetched."""
        requested = set(self.request.symbols)
        succeeded = set(self.series.keys())
        return requested == succeeded and not self.errors

    def missing_symbols(self) -> List[str]:
        """Return list of symbols for which data was not successfully returned."""
        requested = set(self.request.symbols)
        succeeded = set(self.series.keys())
        failed = requested - succeeded
        return sorted(failed)

    def has_partial_success(self) -> bool:
        """Return True if some symbols succeeded and some failed."""
        return bool(self.series) and bool(self.errors)

    class Config:
        validate_assignment = True


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight DTO for "latest quote" use-cases
# ──────────────────────────────────────────────────────────────────────────────


class LatestQuote(BaseModel):
    """
    Simplified quote model for dashboards needing only latest level & change.
    """

    symbol: str = Field(..., description="Ticker symbol")
    price: float = Field(..., description="Last traded or indicative price")
    currency: Optional[str] = Field(
        default=None, description="Currency of price (e.g. EUR)"
    )
    timestamp: datetime = Field(..., description="Timestamp of this quote")
    change: Optional[float] = Field(
        default=None, description="Absolute price change vs previous close"
    )
    change_pct: Optional[float] = Field(
        default=None, description="Percentage price change vs previous close"
    )
    provider: Optional[MarketDataProvider] = Field(
        default=None, description="Data provider"
    )
    source: Optional[Literal["live", "delayed", "end_of_day"]] = Field(
        default="end_of_day", description="Nature of the quote"
    )

    class Config:
        validate_assignment = True
