"""
Fama-French Factor Loader

Utilities to load Fama-French risk factors for use in:

- FactorAnalyzer
- Portfolio attribution
- Backtests

Supports:
- US Fama-French 3-factor (daily)
- US Fama-French 5-factor (daily)
- Europe 3-factor (daily, if available)
- Local CSV files (recommended for robust production)

The remote loader is convenient for research; for production, you should
download & preprocess the CSV once, then point `local_path` to that file.
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


Region = Literal["US", "Europe"]
Frequency = Literal["daily", "monthly"]  # (we implement daily URLs, but design is extendable)
FFFamily = Literal["FF3", "FF5"]


@dataclass
class FamaFrenchConfig:
    """
    Configuration for Fama-French factor loading.

    Attributes:
        region: 'US' or 'Europe' (currently US support is best tested).
        family: 'FF3' (3 factors) or 'FF5' (5 factors).
        frequency: 'daily' or 'monthly' (we focus on daily datasets).
        local_path: Optional path to a pre-cleaned CSV file with factor data.
            If provided, loader will use this instead of remote download.
        use_remote: Whether to attempt remote download if local_path is None.
        convert_to_decimal: Convert percentage returns (100 * R) to decimals.
    """

    region: Region = "US"
    family: FFFamily = "FF5"
    frequency: Frequency = "daily"
    local_path: Optional[str] = None
    use_remote: bool = True
    convert_to_decimal: bool = True


class FamaFrenchLoader:
    """
    Loader for Fama-French factors.

    Typical usage:

    >>> config = FamaFrenchConfig(region="US", family="FF5", frequency="daily")
    >>> loader = FamaFrenchLoader(config)
    >>> df = loader.get_factors(start="2020-01-01", end="2025-01-01")

    Expected output columns (depending on dataset):
        - 'MKT_RF' : Market excess return
        - 'SMB'    : Size factor
        - 'HML'    : Value factor
        - 'RMW'    : Profitability factor (FF5)
        - 'CMA'    : Investment factor (FF5)
        - 'RF'     : Risk-free rate
    """

    def __init__(self, config: Optional[FamaFrenchConfig] = None) -> None:
        self.config = config or FamaFrenchConfig()
        self._cached_factors: Optional[pd.DataFrame] = None

        logger.info(
            "FamaFrenchLoader initialized (region=%s, family=%s, freq=%s, local_path=%s)",
            self.config.region,
            self.config.family,
            self.config.frequency,
            self.config.local_path,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_factors(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load factor time series, optionally filtered by date range.

        Args:
            start: Optional start date (YYYY-MM-DD).
            end: Optional end date (YYYY-MM-DD).

        Returns:
            DataFrame indexed by datetime with factor columns.
        """
        if self._cached_factors is None:
            self._cached_factors = self._load_factors()

        df = self._cached_factors.copy()

        if start is not None:
            df = df[df.index >= pd.to_datetime(start)]
        if end is not None:
            df = df[df.index <= pd.to_datetime(end)]

        return df

    def get_ff3(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        Convenience method: force FF3 family and return.

        (Will not mutate the loader's default config.)
        """
        tmp_cfg = FamaFrenchConfig(
            region=self.config.region,
            family="FF3",
            frequency=self.config.frequency,
            local_path=self.config.local_path,
            use_remote=self.config.use_remote,
            convert_to_decimal=self.config.convert_to_decimal,
        )
        tmp_loader = FamaFrenchLoader(tmp_cfg)
        return tmp_loader.get_factors(start=start, end=end)

    def get_ff5(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """
        Convenience method: force FF5 family and return.

        (Will not mutate the loader's default config.)
        """
        tmp_cfg = FamaFrenchConfig(
            region=self.config.region,
            family="FF5",
            frequency=self.config.frequency,
            local_path=self.config.local_path,
            use_remote=self.config.use_remote,
            convert_to_decimal=self.config.convert_to_decimal,
        )
        tmp_loader = FamaFrenchLoader(tmp_cfg)
        return tmp_loader.get_factors(start=start, end=end)

    # -------------------------------------------------------------------------
    # Internal loading logic
    # -------------------------------------------------------------------------

    def _load_factors(self) -> pd.DataFrame:
        """
        Core loading logic:

        1. If local_path is given → load from local CSV.
        2. Else if use_remote is True → download from Ken French library.
        3. Else → raise ValueError.

        Returns:
            Cleaned factor DataFrame.
        """
        if self.config.local_path:
            logger.info("Loading Fama-French factors from local CSV: %s", self.config.local_path)
            df = self._load_from_local_csv(self.config.local_path)
        elif self.config.use_remote:
            url = self._get_remote_url()
            logger.info("Downloading Fama-French factors from: %s", url)
            df = self._download_and_parse_ken_french_zip(url)
        else:
            raise ValueError("No data source configured: set local_path or use_remote=True")

        if df.empty:
            logger.warning("Loaded Fama-French dataset is empty.")
            return df

        # Standardize column names
        df = self._standardize_columns(df)

        # Convert percent to decimal if needed
        if self.config.convert_to_decimal:
            df = df / 100.0

        # Ensure DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, format="%Y%m%d", errors="coerce")
            df = df.dropna(axis=0, subset=[df.index.name or df.index.names[0] if hasattr(df.index, "names") else None])

        df = df.sort_index()
        logger.info(
            "Loaded Fama-French factors: %d rows, columns=%s",
            len(df),
            list(df.columns),
        )
        return df

    # -------------------------------------------------------------------------
    # Local CSV loading
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_from_local_csv(path: str) -> pd.DataFrame:
        """
        Load factors from a local CSV that already has:
            - a date column (or index) parsable to datetime
            - factor columns (MKT_RF, SMB, HML, ...)

        This is the recommended production approach.

        Args:
            path: Path to CSV file.

        Returns:
            DataFrame indexed by datetime.
        """
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.error("Failed to read local Fama-French CSV '%s': %s", path, exc)
            raise

        # Detect date column
        date_cols = [c for c in df.columns if c.lower() in ("date", "t", "yyyymmdd")]
        if date_cols:
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        else:
            # Assume index already is date-like
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        return df

    # -------------------------------------------------------------------------
    # Remote URL logic
    # -------------------------------------------------------------------------

    def _get_remote_url(self) -> str:
        """
        Return the Ken French library URL for the configured dataset.

        Note:
            These URLs are correct as of typical Ken French naming conventions.
            If they ever change, you can switch to local CSVs instead.
        """
        base = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

        if self.config.region == "US":
            if self.config.family == "FF3":
                # F-F_Research_Data_Factors_daily_CSV.zip
                return f"{base}/F-F_Research_Data_Factors_daily_CSV.zip"
            else:  # FF5
                # F-F_Research_Data_5_Factors_2x3_daily_CSV.zip
                return f"{base}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

        # Europe 3-factor daily (if available)
        if self.config.region == "Europe":
            # Typical name: Europe_3_Factors_Daily_CSV.zip
            # If this URL fails, user should switch to local CSV mode.
            return f"{base}/Europe_3_Factors_Daily_CSV.zip"

        raise ValueError(f"Unsupported region: {self.config.region}")

    # -------------------------------------------------------------------------
    # Ken French ZIP parsing
    # -------------------------------------------------------------------------

    @staticmethod
    def _download_and_parse_ken_french_zip(url: str) -> pd.DataFrame:
        """
        Download and parse a Ken French zip file in CSV format.

        Ken French datasets typically have:
            - Header text lines
            - A line with "Date" header
            - Data rows
            - Blank lines and additional tables (e.g., annual stats) below

        Parsing strategy:
            1. Download ZIP.
            2. Read the single CSV inside (first .CSV entry).
            3. Find the header line with 'Date'.
            4. Read until the first completely blank line after the data.
        """
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            logger.error("Error downloading Ken French data from %s: %s", url, exc)
            raise

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # Get first CSV-like file
            name = None
            for info in zf.infolist():
                if info.filename.lower().endswith((".csv", ".txt")):
                    name = info.filename
                    break
            if name is None:
                raise ValueError("No CSV/TXT file found in Ken French zip archive.")

            raw = zf.read(name).decode("latin1")

        # Split into lines and find where the main table starts/ends
        lines = raw.splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if "Date" in line and ("Mkt-RF" in line or "Mkt-RF" in line.replace(" ", "")):
                header_idx = i
                break

        if header_idx is None:
            # Fallback: try first line starting with a numeric year
            for i, line in enumerate(lines):
                parts = line.split(",")
                if parts and parts[0].strip().isdigit():
                    header_idx = i - 1 if i > 0 else 0
                    break

        if header_idx is None:
            raise ValueError("Could not find header line in Ken French file.")

        # Now capture data lines until a blank line is hit after data starts
        data_lines = [lines[header_idx]]
        for line in lines[header_idx + 1 :]:
            if not line.strip():
                # blank line -> end of table
                break
            data_lines.append(line)

        csv_str = "\n".join(data_lines)

        df = pd.read_csv(io.StringIO(csv_str))
        # Standard name of date column is often "Date" or similar
        if "Date" in df.columns:
            date_col = "Date"
        else:
            # fallback: first column
            date_col = df.columns[0]

        df = df.rename(columns={date_col: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")

        return df

    # -------------------------------------------------------------------------
    # Column standardization
    # -------------------------------------------------------------------------

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to a clean Fama-French convention.

        From Ken French typical labels:
            'Mkt-RF' -> 'MKT_RF'
            'SMB'    -> 'SMB'
            'HML'    -> 'HML'
            'RMW'    -> 'RMW'
            'CMA'    -> 'CMA'
            'RF'     -> 'RF'
        """
        col_map: Dict[str, str] = {}
        for c in df.columns:
            cl = c.strip().upper().replace(" ", "_").replace("-", "_")

            if cl.startswith("MKT_RF") or cl.startswith("MKT__RF") or cl == "MKT_RF":
                col_map[c] = "MKT_RF"
            elif cl.startswith("MKT"):
                col_map[c] = "MKT_RF"
            elif cl.startswith("SMB"):
                col_map[c] = "SMB"
            elif cl.startswith("HML"):
                col_map[c] = "HML"
            elif cl.startswith("RMW"):
                col_map[c] = "RMW"
            elif cl.startswith("CMA"):
                col_map[c] = "CMA"
            elif cl in ("RF", "R_F", "RISK_FREE") or "RF" == cl:
                col_map[c] = "RF"
            else:
                # Keep original if unknown
                col_map[c] = cl

        df = df.rename(columns=col_map)
        return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # This will try to download US FF5 daily from Ken French.
    # For production, you should use local CSV mode with a pre-cleaned file.
    cfg = FamaFrenchConfig(region="US", family="FF5", frequency="daily", use_remote=True)
    loader = FamaFrenchLoader(cfg)

    try:
        factors = loader.get_factors(start="2020-01-01", end=datetime.today().strftime("%Y-%m-%d"))
        print("\nFama-French factors (head):")
        print(factors.head())
        print("\nColumns:", list(factors.columns))
    except Exception as e:
        print(f"\n[ERROR] Could not load remote Fama-French data: {e}")
        print("Hint: For production, download the CSV manually and set FamaFrenchConfig.local_path.")
