"""
Date Utilities Module

Provides robust trading-day utilities used across the JPMorgan
European Equity Dashboard:

- Trading day checks
- Business day offsets
- Quarter boundaries
- Market-hour checks
- EU/US market holiday support
- Clean date handling for backtesting, data pipelines, and plotting

Author: V.L. Siddarth (NexusNext / JPMorganChase Project)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional

import pandas as pd

try:
    # Optional: work with full holiday calendars
    import pandas_market_calendars as mcal
    _HAS_MCAL = True
except Exception:
    _HAS_MCAL = False

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
#  Utility Class
# -----------------------------------------------------------------------------

class DateUtils:
    """
    Institution-grade date and trading-day utility toolkit.

    Designed to work even without pandas_market_calendars installed.
    """

    # Default: European market calendar (XETR)
    MARKET_CODE = "XETR"

    @staticmethod
    def today() -> date:
        """Return today's date (UTC)."""
        return datetime.utcnow().date()

    @staticmethod
    def now() -> datetime:
        """Return current UTC datetime."""
        return datetime.utcnow()

    # -----------------------------------------------------------------------------
    # Trading Day Utilities
    # -----------------------------------------------------------------------------

    @staticmethod
    def is_business_day(d: date | datetime) -> bool:
        """
        True if the given date is a weekday (Mon-Fri).
        Does *not* check holidays unless mcal is available.
        """
        d = DateUtils._to_date(d)
        return d.weekday() < 5  # Monday=0

    @staticmethod
    def is_trading_day(d: date | datetime) -> bool:
        """
        True if market is open on this date.
        Requires pandas_market_calendars for full support.
        Falls back to weekday check.
        """
        d = DateUtils._to_date(d)

        if not _HAS_MCAL:
            return DateUtils.is_business_day(d)

        try:
            cal = mcal.get_calendar(DateUtils.MARKET_CODE)
            schedule = cal.schedule(start_date=d, end_date=d)
            return not schedule.empty
        except Exception:
            return DateUtils.is_business_day(d)

    @staticmethod
    def next_trading_day(d: date | datetime) -> date:
        """Return the next trading day after the given date."""
        d = DateUtils._to_date(d)
        while True:
            d += timedelta(days=1)
            if DateUtils.is_trading_day(d):
                return d

    @staticmethod
    def previous_trading_day(d: date | datetime) -> date:
        """Return the previous trading day before the given date."""
        d = DateUtils._to_date(d)
        while True:
            d -= timedelta(days=1)
            if DateUtils.is_trading_day(d):
                return d

    @staticmethod
    def add_business_days(d: date, n: int) -> date:
        """Add n business days to a date."""
        step = 1 if n > 0 else -1
        count = abs(n)
        while count:
            d += timedelta(days=step)
            if DateUtils.is_business_day(d):
                count -= 1
        return d

    # -----------------------------------------------------------------------------
    # Market Hours
    # -----------------------------------------------------------------------------

    @staticmethod
    def is_market_open(dt: datetime) -> bool:
        """
        True if market is currently open.
        For XETR (German exchange), approximate hours:
            09:00–17:30 CET
        """
        dt = DateUtils._to_datetime(dt)

        if not DateUtils.is_trading_day(dt.date()):
            return False

        # Convert to CET from UTC
        hour = dt.hour + 1  # very rough CET conversion (no DST)
        minute = dt.minute

        open_time = 9 * 60
        close_time = 17 * 60 + 30
        now_minutes = hour * 60 + minute

        return open_time <= now_minutes <= close_time

    # -----------------------------------------------------------------------------
    # Quarter & Year Utilities
    # -----------------------------------------------------------------------------

    @staticmethod
    def get_quarter(d: date | datetime) -> int:
        """Return financial quarter (1–4)."""
        d = DateUtils._to_date(d)
        return (d.month - 1) // 3 + 1

    @staticmethod
    def quarter_start(d: date | datetime) -> date:
        """Return the first day of the quarter."""
        d = DateUtils._to_date(d)
        q = DateUtils.get_quarter(d)
        start_month = (q - 1) * 3 + 1
        return date(d.year, start_month, 1)

    @staticmethod
    def quarter_end(d: date | datetime) -> date:
        """Return the last day of the quarter."""
        d = DateUtils._to_date(d)
        q = DateUtils.get_quarter(d)
        end_month = q * 3
        last_day = DateUtils._last_day_of_month(date(d.year, end_month, 1))
        return last_day

    @staticmethod
    def year_start(year: int) -> date:
        return date(year, 1, 1)

    @staticmethod
    def year_end(year: int) -> date:
        return date(year, 12, 31)

    # -----------------------------------------------------------------------------
    # Date Range Helpers
    # -----------------------------------------------------------------------------

    @staticmethod
    def trading_day_range(start: date | datetime, end: date | datetime) -> List[date]:
        """
        Return a list of all trading days between two dates.
        """
        start = DateUtils._to_date(start)
        end = DateUtils._to_date(end)

        days = pd.date_range(start, end, freq="D")
        return [d.date() for d in days if DateUtils.is_trading_day(d.date())]

    @staticmethod
    def business_day_range(start: date | datetime, end: date | datetime) -> List[date]:
        """Return a list of business days (Mon-Fri) between start and end."""
        start = DateUtils._to_date(start)
        end = DateUtils._to_date(end)

        return [
            (start + timedelta(days=i))
            for i in range((end - start).days + 1)
            if DateUtils.is_business_day(start + timedelta(days=i))
        ]

    # -----------------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------------

    @staticmethod
    def _to_date(d) -> date:
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, date):
            return d
        raise TypeError("Expected date or datetime")

    @staticmethod
    def _to_datetime(d) -> datetime:
        if isinstance(d, datetime):
            return d
        if isinstance(d, date):
            return datetime(d.year, d.month, d.day)
        raise TypeError("Expected date or datetime")

    @staticmethod
    def _last_day_of_month(d: date) -> date:
        """
        Return the last day of the month for the given date.
        """
        if d.month == 12:
            return date(d.year, 12, 31)
        next_month = date(d.year, d.month + 1, 1)
        return next_month - timedelta(days=1)


# -----------------------------------------------------------------------------
# Self-test (only runs when executed directly)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Today:", DateUtils.today())
    print("Is Trading Day:", DateUtils.is_trading_day(DateUtils.today()))
    print("Next Trading Day:", DateUtils.next_trading_day(DateUtils.today()))
    print("Quarter Start:", DateUtils.quarter_start(DateUtils.today()))
    print("Quarter End:", DateUtils.quarter_end(DateUtils.today()))
    print("Trading days this month:",
          len(DateUtils.trading_day_range(date(2025, 1, 1), date(2025, 1, 31))))
