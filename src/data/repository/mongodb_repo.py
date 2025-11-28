# src/data/repository/mongodb_repo.py

"""
MongoDB Repository Layer
------------------------

Production-ready repository for persisting:

- Market Data (OHLCV bars)
- Portfolio snapshots
- Signals
- Analytics outputs

Implements:
- Safe writes with index creation
- Query helpers
- Upsert operations
- Enterprise-grade error handling
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from pymongo import MongoClient, ASCENDING, DESCENDING, errors
from pydantic import BaseModel

from src.data.models.market_data import HistoricalPriceSeries, PriceBar
from src.data.models.portfolio import (
    PortfolioSnapshot,
    PortfolioPerformanceSeries,
    PortfolioPerformancePoint,
)
from src.data.models.signals import SignalPoint, SymbolSignalHistory

logger = logging.getLogger(__name__)


class MongoDBRepository:
    """
    MongoDB repository wrapper for prod-grade financial data storage.

    Collections:
    • market_data
    • signals
    • portfolios
    • performance
    """

    def __init__(
        self,
        uri: str,
        db_name: str = "jpm_dashboard",
        timeout_ms: int = 3000,
    ) -> None:
        try:
            self.client = MongoClient(
                uri,
                serverSelectionTimeoutMS=timeout_ms,
            )
            self.db = self.client[db_name]
            self._initialize_collections()
            logger.info("MongoDBRepository initialized for DB: %s", db_name)
        except Exception as e:
            logger.exception("Failed to initialize MongoDBRepository: %s", e)
            raise

    # -------------------------------------------------------------------------
    # INDEXING
    # -------------------------------------------------------------------------
    def _initialize_collections(self) -> None:
        """Create indexes for high-performance queries."""
        try:
            # Market data
            self.db.market_data.create_index(
                [("symbol", ASCENDING), ("timestamp", ASCENDING)],
                unique=True,
                name="idx_symbol_timestamp",
            )

            # Signals
            self.db.signals.create_index(
                [
                    ("symbol", ASCENDING),
                    ("signal_name", ASCENDING),
                    ("as_of", ASCENDING),
                ],
                unique=True,
                name="idx_signal_unique",
            )

            # Portfolio snapshots
            self.db.portfolios.create_index(
                [("portfolio_id", ASCENDING), ("as_of", DESCENDING)],
                name="idx_portfolio_snapshots",
            )

            # Performance series
            self.db.performance.create_index(
                [("portfolio_id", ASCENDING), ("date", ASCENDING)],
                unique=True,
                name="idx_performance",
            )

        except Exception:
            logger.exception("Error creating MongoDB indexes")
            raise

    # -------------------------------------------------------------------------
    # MARKET DATA
    # -------------------------------------------------------------------------
    def save_price_series(self, series: HistoricalPriceSeries) -> None:
        """Persist HistoricalPriceSeries into MongoDB (flattened schema)."""
        try:
            for bar in series.bars:
                document = {
                    "symbol": series.symbol,
                    "interval": series.interval.value,
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "adj_close": bar.adj_close,
                    "volume": bar.volume,
                    "provider": bar.provider.value,
                }
                self.db.market_data.update_one(
                    {
                        "symbol": series.symbol,
                        "timestamp": bar.timestamp,
                    },
                    {"$set": document},
                    upsert=True,
                )
        except errors.PyMongoError as e:
            logger.exception("Failed to save price series for %s: %s", series.symbol, e)
            raise

    def load_price_series(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> HistoricalPriceSeries:
        """Load OHLCV data and convert to HistoricalPriceSeries."""
        query: Dict[str, Any] = {"symbol": symbol}

        if start:
            query["timestamp"] = {"$gte": start}
        if end:
            query.setdefault("timestamp", {})["$lte"] = end

        try:
            cursor = (
                self.db.market_data.find(query)
                .sort("timestamp", ASCENDING)
            )
            bars = [
                PriceBar(
                    timestamp=doc["timestamp"],
                    open=doc.get("open"),
                    high=doc.get("high"),
                    low=doc.get("low"),
                    close=doc.get("close"),
                    adj_close=doc.get("adj_close"),
                    volume=doc.get("volume"),
                    provider=doc.get("provider"),
                )
                for doc in cursor
            ]
            return HistoricalPriceSeries(symbol=symbol, bars=bars)

        except errors.PyMongoError as e:
            logger.exception("Failed to load price series for %s: %s", symbol, e)
            raise

    # -------------------------------------------------------------------------
    # SIGNALS
    # -------------------------------------------------------------------------
    def save_signal(self, signal: SignalPoint) -> None:
        """Save SignalPoint to database."""
        try:
            doc = signal.dict()
            self.db.signals.update_one(
                {
                    "symbol": signal.symbol,
                    "signal_name": signal.name,
                    "as_of": signal.as_of,
                },
                {"$set": doc},
                upsert=True,
            )
        except errors.PyMongoError:
            logger.exception("Failed to save signal: %s", signal)
            raise

    def load_symbol_signal_history(
        self, symbol: str, signal_name: str
    ) -> SymbolSignalHistory:
        """Load full signal history for a symbol."""
        try:
            cursor = (
                self.db.signals.find(
                    {"symbol": symbol, "signal_name": signal_name}
                )
                .sort("as_of", ASCENDING)
            )
            points = [SignalPoint(**doc) for doc in cursor]
            return SymbolSignalHistory(
                symbol=symbol,
                signal_name=signal_name,
                category=points[0].category if points else None,
                points=points,
            )
        except errors.PyMongoError:
            logger.exception(
                "Failed to load signal history for %s-%s", symbol, signal_name
            )
            raise

    # -------------------------------------------------------------------------
    # PORTFOLIOS
    # -------------------------------------------------------------------------
    def save_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        try:
            self.db.portfolios.update_one(
                {"portfolio_id": snapshot.portfolio_id, "as_of": snapshot.as_of},
                {"$set": snapshot.dict()},
                upsert=True,
            )
        except errors.PyMongoError:
            logger.exception("Failed to save portfolio snapshot: %s", snapshot)
            raise

    def load_portfolio_snapshots(
        self, portfolio_id: str, limit: int = 50
    ) -> List[PortfolioSnapshot]:
        try:
            cursor = (
                self.db.portfolios.find({"portfolio_id": portfolio_id})
                .sort("as_of", DESCENDING)
                .limit(limit)
            )
            return [PortfolioSnapshot(**doc) for doc in cursor]
        except errors.PyMongoError:
            logger.exception("Failed to load snapshots for %s", portfolio_id)
            raise

    # -------------------------------------------------------------------------
    # PERFORMANCE
    # -------------------------------------------------------------------------
    def save_performance_series(
        self, series: PortfolioPerformanceSeries
    ) -> None:
        try:
            for p in series.points:
                self.db.performance.update_one(
                    {"portfolio_id": p.portfolio_id, "date": p.date},
                    {"$set": p.dict()},
                    upsert=True,
                )
        except errors.PyMongoError:
            logger.exception("Failed to save performance for %s", series.portfolio_id)
            raise

    def load_performance_series(
        self, portfolio_id: str
    ) -> PortfolioPerformanceSeries:
        try:
            cursor = (
                self.db.performance.find({"portfolio_id": portfolio_id})
                .sort("date", ASCENDING)
            )
            points = [PortfolioPerformancePoint(**doc) for doc in cursor]
            return PortfolioPerformanceSeries(portfolio_id=portfolio_id, points=points)
        except errors.PyMongoError:
            logger.exception(
                "Failed to load performance series for %s", portfolio_id
            )
            raise
