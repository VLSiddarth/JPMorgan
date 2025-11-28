# src/data/repository/timescale_repo.py

"""
TimescaleDB Repository Layer
----------------------------

Production-ready repository for time-series intensive data:

- High-frequency market data (if needed)
- Factor and risk model time series
- Macro time series

Uses psycopg2 with parameterized queries and basic connection management.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, date

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

import pandas as pd

logger = logging.getLogger(__name__)


class TimescaleRepository:
    """
    TimescaleDB repository wrapper for high-volume time series storage.

    Assumes TimescaleDB is already installed and extension enabled:
        CREATE EXTENSION IF NOT EXISTS timescaledb;
    """

    def __init__(
        self,
        dsn: str,
        schema: str = "public",
        connect_timeout: int = 5,
    ) -> None:
        """
        Args:
            dsn: PostgreSQL/Timescale connection string
            schema: default schema
            connect_timeout: connection timeout in seconds
        """
        try:
            self.dsn = f"{dsn} options='-c search_path={schema}' connect_timeout={connect_timeout}"
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = True
            self.schema = schema
            self._initialize_schema()
            logger.info("TimescaleRepository initialized with schema '%s'", schema)
        except Exception as e:
            logger.exception("Failed to initialize TimescaleRepository: %s", e)
            raise

    # ------------------------------------------------------------------ #
    # Schema & table initialization
    # ------------------------------------------------------------------ #
    def _initialize_schema(self) -> None:
        """Create core hypertables if they do not exist."""
        try:
            with self._conn.cursor() as cur:
                # Example hypertable for macro series
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS macro_series (
                        series_id      TEXT        NOT NULL,
                        ts             TIMESTAMPTZ NOT NULL,
                        value          DOUBLE PRECISION,
                        PRIMARY KEY (series_id, ts)
                    );
                    """
                )
                cur.execute(
                    """
                    SELECT create_hypertable('macro_series', 'ts', if_not_exists => TRUE);
                    """
                )
        except Exception:
            logger.exception("Error initializing Timescale schema")
            raise

    # ------------------------------------------------------------------ #
    # Macro series
    # ------------------------------------------------------------------ #
    def upsert_macro_series(
        self,
        series_id: str,
        df: pd.DataFrame,
        ts_col: str = "ts",
        value_col: str = "value",
    ) -> None:
        """
        Upsert macro time series into macro_series hypertable.

        Args:
            series_id: identifier (e.g., 'EU_HICP_HEADLINE')
            df: DataFrame with datetime index or 'ts' column, and 'value' column
            ts_col: timestamp column name if not using index
            value_col: value column name
        """
        if df is None or df.empty:
            logger.warning("upsert_macro_series called with empty DataFrame")
            return

        if not isinstance(df.index, pd.DatetimeIndex):
            if ts_col not in df.columns:
                raise ValueError(
                    "DataFrame must have DatetimeIndex or a 'ts' column"
                )
            ts = pd.to_datetime(df[ts_col])
        else:
            ts = df.index

        values = df[value_col].astype(float)

        records = [
            (series_id, ts_i.to_pydatetime(), float(val))
            for ts_i, val in zip(ts, values)
            if pd.notna(val)
        ]

        if not records:
            logger.warning("No non-NaN values to upsert for series_id=%s", series_id)
            return

        insert_sql = """
            INSERT INTO macro_series (series_id, ts, value)
            VALUES %s
            ON CONFLICT (series_id, ts)
            DO UPDATE SET value = EXCLUDED.value;
        """
        try:
            with self._conn.cursor() as cur:
                execute_values(cur, insert_sql, records)
        except Exception:
            logger.exception(
                "Failed to upsert macro series for series_id=%s", series_id
            )
            raise

    def load_macro_series(
        self,
        series_id: str,
        start: Optional[datetime | date] = None,
        end: Optional[datetime | date] = None,
    ) -> pd.Series:
        """
        Load macro series as pandas Series.

        Args:
            series_id: series identifier
            start: optional start datetime/date
            end: optional end datetime/date

        Returns:
            pandas.Series indexed by timestamp
        """
        query = ["SELECT ts, value FROM macro_series WHERE series_id = %s"]
        params: List[Any] = [series_id]

        if start is not None:
            query.append("AND ts >= %s")
            params.append(start)
        if end is not None:
            query.append("AND ts <= %s")
            params.append(end)

        query.append("ORDER BY ts ASC")
        sql = " ".join(query)

        try:
            with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                if not rows:
                    return pd.Series(dtype=float)

                idx = [r["ts"] for r in rows]
                vals = [r["value"] for r in rows]
                return pd.Series(vals, index=pd.to_datetime(idx), name=series_id)
        except Exception:
            logger.exception("Failed to load macro series for %s", series_id)
            raise

    def close(self) -> None:
        """Close underlying DB connection."""
        try:
            if self._conn:
                self._conn.close()
        except Exception:
            logger.exception("Error closing Timescale connection")
