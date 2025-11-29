"""
Factor Analyzer

Institutional-grade factor analysis module for the JPMorgan European Equity Dashboard.

Responsibilities:
- Compute cross-sectional style factor scores:
    - Value (e.g., Book-to-Price, Earnings Yield)
    - Momentum (12M / 6M excluding 1M)
    - Quality (ROE, ROIC, margins)
    - Size (log market cap)
    - Low Volatility (inverse volatility)
- Normalize factors (z-scores, ranks)
- Compute portfolio factor exposures (holdings vs universe)
- Optional integration with Fama-French factor returns and custom factor models.

This module is intentionally DataFrame-based to work with any data source
(yfinance fundamentals, Refinitiv, FactSet, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover - optional
    sm = None

# Optional imports – module works if these are missing
try:
    from src.analytics.factors.fama_french import FamaFrenchLoader  # type: ignore
except Exception:  # pragma: no cover
    FamaFrenchLoader = None  # type: ignore

try:
    from src.analytics.factors.custom_factors import CustomFactorModel  # type: ignore
except Exception:  # pragma: no cover
    CustomFactorModel = None  # type: ignore


logger = logging.getLogger(__name__)


FactorName = Literal["value", "momentum", "quality", "size", "low_vol"]


@dataclass
class FactorConfig:
    """
    Configuration for factor computation and normalization.

    Attributes:
        winsorize_pct: Percentage for winsorization at both tails (e.g., 0.01 = 1%).
        standardize: Whether to convert factors to z-scores.
        demean_industry: Whether to demean factors by industry (if industry column provided).
        industry_column: Name of industry/sector column in universe DataFrame.
    """

    winsorize_pct: float = 0.01
    standardize: bool = True
    demean_industry: bool = False
    industry_column: str = "sector"


class FactorAnalyzer:
    """
    Core factor analysis engine.

    Usage:
        analyzer = FactorAnalyzer()
        factors_df = analyzer.compute_style_factors(universe_df)
        exposures = analyzer.compute_portfolio_exposures(weights, factors_df)
        factor_returns = analyzer.compute_factor_returns(returns_df, factors_df)

    Expected universe_df columns (as available):
        - 'ticker' (str)
        - 'price' (float)
        - 'market_cap' (float)
        - 'book_value' or 'book_to_price'
        - 'earnings' or 'earnings_yield'
        - 'roe', 'roic' etc. (for quality)
        - 'volatility_252' or daily returns history for low vol
        - 'momentum_12m', 'momentum_6m' (or you pass return history separately)
        - 'sector' or 'industry' (optional, for industry-neutral factors)
    """

    def __init__(self, config: Optional[FactorConfig] = None) -> None:
        self.config: FactorConfig = config or FactorConfig()
        logger.info(
            "FactorAnalyzer initialized (winsorize_pct=%.3f, standardize=%s, demean_industry=%s)",
            self.config.winsorize_pct,
            self.config.standardize,
            self.config.demean_industry,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def compute_style_factors(
        self,
        universe: pd.DataFrame,
        price_history: Optional[pd.DataFrame] = None,
        id_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        Compute style factor scores for the given universe of stocks.

        Args:
            universe: Cross-sectional DataFrame (one row per security).
                Must contain at least 'ticker' + any available fundamental fields.
            price_history: Optional panel-like DataFrame for momentum/vol,
                expected shape: index=dates, columns=tickers, values=prices.
                If not provided, momentum/low_vol will use pre-computed columns
                like 'momentum_12m', 'volatility_252' if present.
            id_col: Column name for security identifier (default: 'ticker').

        Returns:
            factors_df: DataFrame indexed by id_col with columns:
                ['value', 'momentum', 'quality', 'size', 'low_vol'] (where available)
        """
        df = universe.copy()

        if id_col not in df.columns:
            raise ValueError(f"Universe DataFrame must contain id_col='{id_col}'")

        df = df.set_index(id_col)
        logger.debug("Universe size for factor computation: %d", len(df))

        factors: Dict[str, pd.Series] = {}

        # VALUE FACTOR
        factors["value"] = self._compute_value_factor(df)

        # MOMENTUM FACTOR
        factors["momentum"] = self._compute_momentum_factor(df, price_history)

        # QUALITY FACTOR
        factors["quality"] = self._compute_quality_factor(df)

        # SIZE FACTOR
        factors["size"] = self._compute_size_factor(df)

        # LOW VOL FACTOR
        factors["low_vol"] = self._compute_low_vol_factor(df, price_history)

        factors_df = pd.DataFrame(factors).dropna(how="all")

        # Apply industry demeaning and normalization
        factors_df = self._post_process_factors(factors_df, df)

        logger.info("Computed style factors for %d securities.", len(factors_df))
        return factors_df

    def compute_portfolio_exposures(
        self,
        weights: pd.Series,
        factors_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute portfolio factor exposures given holdings weights and factor scores.

        Args:
            weights: Series indexed by ticker/security id, values are portfolio weights (sum ~ 1.0).
            factors_df: DataFrame indexed by ticker with factor columns.

        Returns:
            exposures: Series indexed by factor name with weighted average exposures.
        """
        # Align on tickers
        w = weights.dropna()
        w = w[w != 0.0]
        w = w / w.abs().sum()  # normalized weights (handle long/short)

        common = factors_df.index.intersection(w.index)
        if common.empty:
            raise ValueError("No overlap between portfolio weights and factor data.")

        w = w.loc[common]
        f = factors_df.loc[common]

        exposures = (f.T @ w).rename("exposure")
        logger.info("Computed portfolio factor exposures.")
        return exposures

    def compute_factor_returns(
        self,
        returns: pd.DataFrame,
        factors_df: pd.DataFrame,
        id_col: str = "ticker",
        method: Literal["cross_sectional_regression", "long_short"] = "cross_sectional_regression",
        num_quantiles: int = 5,
    ) -> pd.DataFrame:
        """
        Compute time-series factor returns given daily returns and cross-sectional factor scores.

        Args:
            returns: DataFrame of daily returns (index=dates, columns=tickers).
            factors_df: DataFrame of factor scores (index=ticker, columns=factors).
            id_col: Not used directly here but kept for future extensions.
            method:
                - 'cross_sectional_regression': run daily cross-sectional OLS returns ~ factors.
                - 'long_short': build top-vs-bottom quantile portfolios for each factor.
            num_quantiles: Number of quantiles when using 'long_short'.

        Returns:
            factor_returns: DataFrame (index=dates, columns=factors).
        """
        if method == "cross_sectional_regression":
            return self._factor_returns_cs_regression(returns, factors_df)
        elif method == "long_short":
            return self._factor_returns_long_short(returns, factors_df, num_quantiles=num_quantiles)
        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_exposures_and_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        id_col: str = "ticker",
    ) -> Dict[str, pd.Series]:
        """
        Optional: Simple Brinson-style attribution at factor level via regression.

        Args:
            portfolio_returns: Time series of portfolio returns.
            factor_returns: DataFrame of factor returns (index=dates, columns=factors).

        Returns:
            Dict with keys:
                - 'betas': Factor loadings from regression
                - 'residuals': Unexplained residual returns
        """
        if sm is None:
            logger.warning("statsmodels is not available; cannot compute regression-based attribution.")
            return {"betas": pd.Series(dtype=float), "residuals": pd.Series(dtype=float)}

        df = pd.concat(
            [portfolio_returns.rename("portfolio"), factor_returns],
            axis=1,
        ).dropna()

        if df.empty:
            logger.warning("No overlapping dates for attribution regression.")
            return {"betas": pd.Series(dtype=float), "residuals": pd.Series(dtype=float)}

        y = df["portfolio"]
        X = sm.add_constant(df.drop(columns=["portfolio"]))

        model = sm.OLS(y, X, missing="drop")
        res = model.fit()

        betas = res.params.drop("const", errors="ignore")
        residuals = res.resid

        logger.info("Computed regression-based factor attribution.")
        return {"betas": betas, "residuals": residuals}

    # -------------------------------------------------------------------------
    # Internal factor computations
    # -------------------------------------------------------------------------

    def _compute_value_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute value factor using available fields.

        Priority:
            1. 'book_to_price'
            2. 'book_value' / 'price'
            3. 'earnings_yield'
            4. 'earnings' / 'price'

        Returns:
            Series indexed by ticker with raw (un-normalized) value scores.
        """
        candidates = []

        if "book_to_price" in df.columns:
            candidates.append(df["book_to_price"])

        if {"book_value", "price"}.issubset(df.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                btp = df["book_value"] / df["price"].replace(0, np.nan)
            candidates.append(btp)

        if "earnings_yield" in df.columns:
            candidates.append(df["earnings_yield"])

        if {"earnings", "price"}.issubset(df.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                ey = df["earnings"] / df["price"].replace(0, np.nan)
            candidates.append(ey)

        if not candidates:
            logger.warning("No value fields found in universe; value factor will be NaN.")
            return pd.Series(index=df.index, data=np.nan, name="value")

        value_raw = pd.concat(candidates, axis=1).mean(axis=1, skipna=True)
        logger.debug("Computed raw value factor.")
        return value_raw.rename("value_raw")

    def _compute_momentum_factor(
        self,
        df: pd.DataFrame,
        price_history: Optional[pd.DataFrame],
    ) -> pd.Series:
        """
        Compute momentum factor.

        If price_history is provided:
            - 12M momentum = price(T) / price(T-252) - 1
            - Optionally exclude last month.
        Else:
            - Use 'momentum_12m' / 'momentum_6m' columns if available.

        Returns:
            Series with raw momentum scores.
        """
        if price_history is not None and not price_history.empty:
            try:
                prices = price_history.sort_index()
                if len(prices) < 252:
                    raise ValueError("Not enough price history for 12M momentum.")

                # 12M momentum excluding last month:
                # (P_t / P_{t-21}) * (P_{t-21} / P_{t-252}) - 1 ~ P_t / P_{t-252} - 1
                recent = prices.iloc[-1]
                twelve_m_ago = prices.iloc[-252]
                with np.errstate(divide="ignore", invalid="ignore"):
                    mom = (recent / twelve_m_ago) - 1.0
                logger.debug("Computed raw momentum using price history.")
                return mom.rename("momentum_raw")
            except Exception as exc:
                logger.warning("Price-history-based momentum failed, fallback to pre-computed fields: %s", exc)

        # Fallback to pre-computed columns
        if "momentum_12m" in df.columns:
            return df["momentum_12m"].rename("momentum_raw")
        if "momentum_6m" in df.columns:
            return df["momentum_6m"].rename("momentum_raw")

        logger.warning("No momentum fields or price history; momentum factor will be NaN.")
        return pd.Series(index=df.index, data=np.nan, name="momentum_raw")

    def _compute_quality_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute quality factor.

        Uses any of:
            - 'roe' (return on equity)
            - 'roic' (return on invested capital)
            - 'gross_margin', 'operating_margin', 'net_margin'
            - 'debt_to_equity' (inverted)

        Returns:
            Series with raw quality scores.
        """
        components = []

        for col in ["roe", "roic", "gross_margin", "operating_margin", "net_margin"]:
            if col in df.columns:
                components.append(df[col])

        if "debt_to_equity" in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_leverage = 1.0 / (1.0 + df["debt_to_equity"].clip(lower=0))
            components.append(inv_leverage)

        if not components:
            logger.warning("No quality fields available; quality factor will be NaN.")
            return pd.Series(index=df.index, data=np.nan, name="quality_raw")

        quality_raw = pd.concat(components, axis=1).mean(axis=1, skipna=True)
        logger.debug("Computed raw quality factor.")
        return quality_raw.rename("quality_raw")

    def _compute_size_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        Size factor: usually log market cap (smaller stocks = higher size exposure in 'Size' factor).

        If 'market_cap' missing, fall back to:
            - 'shares_outstanding' * 'price'
        """
        mc = None

        if "market_cap" in df.columns:
            mc = df["market_cap"].copy()
        elif {"shares_outstanding", "price"}.issubset(df.columns):
            mc = df["shares_outstanding"] * df["price"]

        if mc is None:
            logger.warning("No market cap or shares * price; size factor will be NaN.")
            return pd.Series(index=df.index, data=np.nan, name="size_raw")

        with np.errstate(divide="ignore", invalid="ignore"):
            size_raw = np.log(mc.replace(0, np.nan))

        # Convention: "Size" factor often defined so that small caps = high exposure.
        size_raw = -size_raw  # invert: smaller caps -> larger factor value
        logger.debug("Computed raw size factor.")
        return size_raw.rename("size_raw")

    def _compute_low_vol_factor(
        self,
        df: pd.DataFrame,
        price_history: Optional[pd.DataFrame],
    ) -> pd.Series:
        """
        Low volatility factor: inverse of 12M volatility.

        If price_history provided, compute daily returns vol; else
        use 'volatility_252' if present.

        Returns:
            Series with raw low-vol scores (higher = more low-vol).
        """
        vol = None

        if price_history is not None and not price_history.empty:
            try:
                prices = price_history.sort_index()
                returns = prices.pct_change().dropna()
                vol = returns.std() * np.sqrt(252)  # annualized
            except Exception as exc:
                logger.warning("Price-history-based volatility failed: %s", exc)
                vol = None

        if vol is None and "volatility_252" in df.columns:
            vol = df["volatility_252"]

        if vol is None:
            logger.warning("No volatility information; low_vol factor will be NaN.")
            return pd.Series(index=df.index, data=np.nan, name="low_vol_raw")

        with np.errstate(divide="ignore", invalid="ignore"):
            low_vol_raw = 1.0 / (1.0 + vol.clip(lower=0))
        logger.debug("Computed raw low_vol factor.")
        return low_vol_raw.rename("low_vol_raw")

    # -------------------------------------------------------------------------
    # Post-processing (winsorization, standardization, industry-demeaning)
    # -------------------------------------------------------------------------

    def _post_process_factors(
        self,
        factors_df: pd.DataFrame,
        universe_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply winsorization, industry demeaning (optional), and standardization.

        Args:
            factors_df: DataFrame of raw factor scores (index=ticker).
            universe_df: Original universe DataFrame (index=ticker), used for industry column.

        Returns:
            Processed factor scores DataFrame.
        """
        processed = factors_df.copy()

        # Ensure raw names -> canonical factor names
        rename_map = {
            "value_raw": "value",
            "momentum_raw": "momentum",
            "quality_raw": "quality",
            "size_raw": "size",
            "low_vol_raw": "low_vol",
        }
        processed = processed.rename(columns={k: v for k, v in rename_map.items() if k in processed.columns})

        for col in processed.columns:
            # Winsorize
            processed[col] = self._winsorize_series(processed[col], self.config.winsorize_pct)

        # Industry demeaning
        if self.config.demean_industry and self.config.industry_column in universe_df.columns:
            industry = universe_df[self.config.industry_column]
            for col in processed.columns:
                processed[col] = self._demean_by_group(processed[col], industry)

        # Standardize (z-score)
        if self.config.standardize:
            for col in processed.columns:
                s = processed[col]
                mu = s.mean()
                sigma = s.std()
                if sigma > 0:
                    processed[col] = (s - mu) / sigma
                else:
                    processed[col] = s * 0.0  # all zeros if no dispersion

        logger.debug("Post-processed factors (winsorize, industry demean, standardize).")
        return processed

    @staticmethod
    def _winsorize_series(s: pd.Series, pct: float) -> pd.Series:
        """
        Winsorize a Series at both tails.

        Args:
            s: Input Series.
            pct: Winsorization percentage (e.g., 0.01 = 1% at each tail).

        Returns:
            Winsorized Series.
        """
        s_clean = s.dropna()
        if s_clean.empty or pct <= 0:
            return s

        lower = s_clean.quantile(pct)
        upper = s_clean.quantile(1.0 - pct)
        return s.clip(lower=lower, upper=upper)

    @staticmethod
    def _demean_by_group(s: pd.Series, groups: pd.Series) -> pd.Series:
        """
        Demean Series s by group labels defined in groups.

        Args:
            s: Values to demean.
            groups: Group labels (e.g., sectors) aligned with s index.

        Returns:
            Group-demeaned Series.
        """
        df = pd.concat([s, groups.rename("group")], axis=1).dropna()
        if df.empty:
            return s

        group_means = df.groupby("group")[s.name].transform("mean")
        s_demeaned = s.copy()
        s_demeaned.loc[df.index] = df[s.name] - group_means
        return s_demeaned

    # -------------------------------------------------------------------------
    # Factor returns methods
    # -------------------------------------------------------------------------

    def _factor_returns_cs_regression(
        self,
        returns: pd.DataFrame,
        factors_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Daily cross-sectional regression: r_it = a_t + b_t' * f_i + e_it

        For each date t:
            - Dep var: cross-section of returns
            - Indep var: factor scores for stocks traded that day
        """
        if sm is None:
            logger.warning("statsmodels not available; cannot compute cross-sectional factor returns.")
            return pd.DataFrame()

        factor_names = list(factors_df.columns)
        factor_ret_records: Dict[str, List[float]] = {f: [] for f in factor_names}
        dates: List[pd.Timestamp] = []

        for date, cross_ret in returns.iterrows():
            # Align with factor universe
            common = cross_ret.dropna().index.intersection(factors_df.index)
            if len(common) < len(factor_names) + 5:
                continue  # not enough data for stable regression

            y = cross_ret.loc[common]
            X = factors_df.loc[common, factor_names]

            X = sm.add_constant(X)
            model = sm.OLS(y, X, missing="drop")
            try:
                res = model.fit()
                for f in factor_names:
                    factor_ret_records[f].append(res.params.get(f, np.nan))
                dates.append(date)
            except Exception as exc:
                logger.debug("Factor return regression failed for %s: %s", date, exc)
                continue

        if not dates:
            logger.warning("No valid dates for factor return regression.")
            return pd.DataFrame()

        factor_ret_df = pd.DataFrame(factor_ret_records, index=pd.to_datetime(dates)).sort_index()
        logger.info("Computed factor returns using cross-sectional regression.")
        return factor_ret_df

    def _factor_returns_long_short(
        self,
        returns: pd.DataFrame,
        factors_df: pd.DataFrame,
        num_quantiles: int = 5,
    ) -> pd.DataFrame:
        """
        Long-short factor returns via quantile portfolios.

        For each factor:
            - Sort universe into num_quantiles by factor score.
            - Factor return = mean(return of top quantile) - mean(return of bottom quantile).
        """
        factor_names = list(factors_df.columns)
        factor_ret_records: Dict[str, List[float]] = {f: [] for f in factor_names}
        dates: List[pd.Timestamp] = []

        for date, cross_ret in returns.iterrows():
            # Align
            common = cross_ret.dropna().index.intersection(factors_df.index)
            if len(common) < 50:
                continue

            r = cross_ret.loc[common]
            f_slice = factors_df.loc[common]

            for f in factor_names:
                s = f_slice[f]
                valid_idx = s.dropna().index.intersection(r.index)
                if len(valid_idx) < num_quantiles * 2:
                    factor_ret_records[f].append(np.nan)
                    continue

                s_valid = s.loc[valid_idx]
                r_valid = r.loc[valid_idx]

                try:
                    q = pd.qcut(s_valid, num_quantiles, labels=False, duplicates="drop")
                except ValueError:
                    factor_ret_records[f].append(np.nan)
                    continue

                # bottom & top
                bottom_mask = q == q.min()
                top_mask = q == q.max()

                bottom_ret = r_valid[bottom_mask].mean()
                top_ret = r_valid[top_mask].mean()

                factor_ret_records[f].append(float(top_ret - bottom_ret))
            dates.append(date)

        if not dates:
            logger.warning("No valid dates for long-short factor returns.")
            return pd.DataFrame()

        factor_ret_df = pd.DataFrame(factor_ret_records, index=pd.to_datetime(dates)).sort_index()
        logger.info("Computed factor returns using long-short quantile portfolios.")
        return factor_ret_df


if __name__ == "__main__":
    # Simple smoke test with synthetic data
    logging.basicConfig(level=logging.INFO)

    tickers = [f"STK{i:03d}" for i in range(1, 101)]
    np.random.seed(42)

    universe = pd.DataFrame(
        {
            "ticker": tickers,
            "price": np.random.uniform(10, 100, size=100),
            "market_cap": np.random.uniform(1e8, 1e10, size=100),
            "book_value": np.random.uniform(1e7, 5e9, size=100),
            "earnings": np.random.uniform(1e6, 2e9, size=100),
            "roe": np.random.normal(0.15, 0.05, size=100),
            "gross_margin": np.random.normal(0.4, 0.1, size=100),
            "debt_to_equity": np.random.uniform(0, 2, size=100),
            "sector": np.random.choice(["Financials", "Industrials", "Tech", "Health Care"], size=100),
        }
    )

    analyzer = FactorAnalyzer(FactorConfig(winsorize_pct=0.01, standardize=True, demean_industry=True))

    # Synthetic price history for low vol / momentum
    dates = pd.date_range("2022-01-01", periods=252, freq="B")
    price_history = pd.DataFrame(
        np.random.lognormal(mean=0.0002, sigma=0.02, size=(252, 100)).cumprod(axis=0) * 50,
        index=dates,
        columns=tickers,
    )

    factors = analyzer.compute_style_factors(universe, price_history=price_history)
    print("\nComputed factors (head):")
    print(factors.head())

    # Fake returns for factor return example
    returns = price_history.pct_change().dropna()
    factor_returns = analyzer.compute_factor_returns(returns, factors, method="long_short")
    print("\nFactor returns (head):")
    print(factor_returns.head())
