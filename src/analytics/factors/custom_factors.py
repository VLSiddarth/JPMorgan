"""
Custom Factor Models

This module defines a flexible framework for JPMorgan-style
custom factor construction on top of the core style factors.

Use cases:
- Thematic factors (e.g., "GRANOLAS", "EU Defense", "German Fiscal Play")
- ESG-style scores (if you have ESG data)
- Macro-sensitive factors (e.g., China-sensitive exporters)
- Any proprietary scoring models for the European equity universe

Key concepts:
- CustomFactor: single factor definition with a compute() method
- CustomFactorModel: manages multiple custom factors and applies them to universe

All implementations here are compatible with:
- src.analytics.factors.analyzer.FactorAnalyzer
- Portfolio optimizer & attribution modules
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Protocol, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol / Interface for custom factors
# ---------------------------------------------------------------------------

class CustomFactor(Protocol):
    """
    Protocol for custom factor implementations.

    Any custom factor must implement:
        - name: property
        - compute(universe: pd.DataFrame) -> pd.Series
    """

    @property
    def name(self) -> str:  # pragma: no cover - interface
        ...

    def compute(self, universe: pd.DataFrame) -> pd.Series:  # pragma: no cover - interface
        ...


# ---------------------------------------------------------------------------
# Example concrete custom factors
# ---------------------------------------------------------------------------

@dataclass
class BasketMembershipFactor:
    """
    Simple binary or scaled factor based on membership in a defined basket.

    Example: GRANOLAS, EU Defense, German Fiscal Play.

    Attributes:
        name: Factor name (e.g., "granolas_exposure").
        basket_tickers: Set/list of tickers that belong to the basket.
        id_col: Column name for ticker/identifier in universe.
        score_if_member: Score assigned if security is in basket.
        score_if_not: Score assigned if security is not in basket.
    """

    name: str
    basket_tickers: List[str]
    id_col: str = "ticker"
    score_if_member: float = 1.0
    score_if_not: float = 0.0

    def compute(self, universe: pd.DataFrame) -> pd.Series:
        if self.id_col not in universe.columns:
            raise ValueError(f"Universe must contain '{self.id_col}' column for BasketMembershipFactor.")

        basket_set = set(self.basket_tickers)
        s = universe[self.id_col].astype(str).isin(basket_set).astype(float)
        s = s.replace({1.0: self.score_if_member, 0.0: self.score_if_not})
        s.index = universe[self.id_col]
        s.name = self.name

        logger.debug("Computed basket membership factor '%s' for %d securities.", self.name, len(s))
        return s


@dataclass
class ThematicScoreFactor:
    """
    Factor based on a pre-computed thematic score column in the universe.

    Example:
        - 'china_exposure_score'
        - 'green_transition_score'
        - 'ai_exposure_score'

    Attributes:
        name: Factor name.
        source_column: Column in universe that contains the raw score.
        winsorize_pct: Optional winsorization at tails.
        standardize: Whether to z-score the factor.
    """

    name: str
    source_column: str
    winsorize_pct: float = 0.01
    standardize: bool = True

    def compute(self, universe: pd.DataFrame) -> pd.Series:
        if self.source_column not in universe.columns:
            logger.warning(
                "Universe missing column '%s' for ThematicScoreFactor '%s'. Returning NaNs.",
                self.source_column,
                self.name,
            )
            s = pd.Series(index=universe.index, data=np.nan, name=self.name)
            return s

        s = universe[self.source_column].astype(float)

        # Winsorize
        if self.winsorize_pct > 0:
            s = _winsorize_series(s, self.winsorize_pct)

        # Standardize
        if self.standardize:
            mu = s.mean()
            sigma = s.std()
            if sigma > 0:
                s = (s - mu) / sigma
            else:
                s = s * 0.0

        s.name = self.name
        logger.debug("Computed thematic score factor '%s'.", self.name)
        return s


@dataclass
class CompositeFactor:
    """
    Composite custom factor as a weighted combination of other factor columns.

    This can mix:
        - core style factors ('value', 'momentum', ...)
        - other custom factors
        - arbitrary numeric columns from the universe

    Example:
        "europe_quality_value" = 0.6 * quality + 0.4 * value

    Attributes:
        name: Composite factor name.
        components: Dict[column_name, weight].
        source: "factors", "universe", or "both":
            - 'factors' -> looks up in factors_df
            - 'universe' -> looks up in universe
            - 'both' -> tries factors_df first then universe
        standardize: Whether to z-score the composite factor.
    """

    name: str
    components: Dict[str, float]
    source: str = "both"
    standardize: bool = True

    def compute(
        self,
        universe: pd.DataFrame,
        factors_df: Optional[pd.DataFrame] = None,
        id_col: str = "ticker",
    ) -> pd.Series:
        if id_col not in universe.columns:
            raise ValueError(f"Universe must contain id_col '{id_col}' for CompositeFactor.")

        universe_indexed = universe.set_index(id_col)
        # Start with zero series
        composite = pd.Series(index=universe_indexed.index, data=0.0, dtype=float)

        for col, weight in self.components.items():
            source_series = None

            if self.source in ("factors", "both") and factors_df is not None and col in factors_df.columns:
                source_series = factors_df[col]
            elif self.source in ("universe", "both") and col in universe_indexed.columns:
                source_series = universe_indexed[col]

            if source_series is None:
                logger.warning(
                    "CompositeFactor '%s' component '%s' not found in selected sources. Skipping.",
                    self.name,
                    col,
                )
                continue

            aligned = source_series.reindex(composite.index)
            composite = composite.add(weight * aligned, fill_value=0.0)

        if self.standardize and composite.notna().any():
            mu = composite.mean()
            sigma = composite.std()
            if sigma > 0:
                composite = (composite - mu) / sigma
            else:
                composite = composite * 0.0

        composite.name = self.name
        logger.debug("Computed composite factor '%s'.", self.name)
        return composite


# ---------------------------------------------------------------------------
# CustomFactorModel – manager for multiple custom factors
# ---------------------------------------------------------------------------

@dataclass
class CustomFactorModel:
    """
    Manager for custom factors.

    Responsibilities:
        - Register custom factor objects.
        - Compute all registered factors for a given universe.
        - Merge custom factors into existing factor DataFrame.

    Typical usage:

        model = CustomFactorModel()

        model.register_factor(
            BasketMembershipFactor(
                name="granolas_exposure",
                basket_tickers=[...],
                id_col="ticker"
            )
        )

        custom_factors = model.compute_all(universe, base_factors_df=factors_df)
    """

    factors: Dict[str, CustomFactor] = field(default_factory=dict)

    def register_factor(self, factor: CustomFactor) -> None:
        """
        Register a new custom factor.

        Args:
            factor: Any object implementing the CustomFactor protocol.
        """
        if factor.name in self.factors:
            logger.warning("Overwriting existing custom factor '%s'.", factor.name)
        self.factors[factor.name] = factor
        logger.info("Registered custom factor '%s'.", factor.name)

    def remove_factor(self, name: str) -> None:
        """
        Remove a custom factor by name (if exists)."""
        if name in self.factors:
            del self.factors[name]
            logger.info("Removed custom factor '%s'.", name)
        else:
            logger.warning("Attempted to remove non-existent custom factor '%s'.", name)

    def list_factors(self) -> List[str]:
        """
        List names of registered custom factors."""
        return list(self.factors.keys())

    def compute_all(
        self,
        universe: pd.DataFrame,
        base_factors_df: Optional[pd.DataFrame] = None,
        id_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        Compute all registered custom factors for a given universe.

        Args:
            universe: Cross-sectional DataFrame of securities.
            base_factors_df: Optional DataFrame of existing factors for composite use.
            id_col: Identifier column (default 'ticker').

        Returns:
            DataFrame of custom factor scores (index=ticker, columns=factor names).
        """
        if id_col not in universe.columns:
            raise ValueError(f"Universe must contain id_col '{id_col}' for CustomFactorModel.")

        universe_indexed = universe.set_index(id_col)

        result: Dict[str, pd.Series] = {}

        for name, factor in self.factors.items():
            try:
                if isinstance(factor, CompositeFactor):
                    s = factor.compute(universe, base_factors_df, id_col=id_col)
                else:
                    s = factor.compute(universe)
                    # Align to index=ticker
                    if s.index.equals(universe_indexed.index) is False:
                        # If series indexed by something else (like integer), reindex by tickers
                        s.index = universe_indexed.index
                result[name] = s
            except Exception as exc:
                logger.error("Error computing custom factor '%s': %s", name, exc, exc_info=True)
                # Fail-safe: return NaNs for that factor
                result[name] = pd.Series(index=universe_indexed.index, data=np.nan, name=name)

        if not result:
            logger.warning("No custom factors registered. Returning empty DataFrame.")
            return pd.DataFrame(index=universe_indexed.index)

        df_custom = pd.DataFrame(result)
        logger.info("Computed %d custom factors for %d securities.", len(df_custom.columns), len(df_custom))
        return df_custom

    def merge_with_base_factors(
        self,
        base_factors_df: pd.DataFrame,
        custom_factors_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge base (style) factors and custom factors into one DataFrame.

        Args:
            base_factors_df: DataFrame of style/core factors.
            custom_factors_df: DataFrame of custom factors (same index).

        Returns:
            Combined factors DataFrame.
        """
        combined = base_factors_df.copy()
        for col in custom_factors_df.columns:
            if col in combined.columns:
                logger.warning(
                    "Custom factor '%s' already exists in base factors. Overwriting.",
                    col,
                )
            combined[col] = custom_factors_df[col]

        logger.info(
            "Merged base factors (%d cols) with custom factors (%d cols) -> total %d cols.",
            base_factors_df.shape[1],
            custom_factors_df.shape[1],
            combined.shape[1],
        )
        return combined


# ---------------------------------------------------------------------------
# Utility function for winsorization (shared with ThematicScoreFactor)
# ---------------------------------------------------------------------------

def _winsorize_series(s: pd.Series, pct: float) -> pd.Series:
    """
    Winsorize a Series at both tails.

    Args:
        s: Input Series.
        pct: Winsorization percentage per tail (e.g., 0.01 = 1%).

    Returns:
        Winsorized Series.
    """
    s_clean = s.dropna()
    if s_clean.empty or pct <= 0:
        return s

    lower = s_clean.quantile(pct)
    upper = s_clean.quantile(1.0 - pct)
    return s.clip(lower=lower, upper=upper)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tickers = [f"STK{i:03d}" for i in range(1, 21)]
    np.random.seed(42)

    # Synthetic universe
    universe = pd.DataFrame(
        {
            "ticker": tickers,
            "price": np.random.uniform(10, 100, size=20),
            "market_cap": np.random.uniform(1e8, 1e10, size=20),
            "china_exposure_score": np.random.uniform(-1, 1, size=20),
        }
    )

    # Example base factors
    base_factors_df = pd.DataFrame(
        {
            "value": np.random.normal(0, 1, size=20),
            "momentum": np.random.normal(0, 1, size=20),
            "quality": np.random.normal(0, 1, size=20),
        },
        index=universe["ticker"],
    )

    # Example GRANOLAS basket
    granolas = [
        "STK001",
        "STK003",
        "STK005",
        "STK007",
        "STK009",
        "STK011",
        "STK013",
        "STK015",
        "STK017",
        "STK019",
        "STK020",
    ]

    model = CustomFactorModel()

    # Basket factor
    model.register_factor(
        BasketMembershipFactor(
            name="granolas_exposure",
            basket_tickers=granolas,
            id_col="ticker",
            score_if_member=1.0,
            score_if_not=0.0,
        )
    )

    # Thematic factor
    model.register_factor(
        ThematicScoreFactor(
            name="china_sensitivity",
            source_column="china_exposure_score",
            winsorize_pct=0.01,
            standardize=True,
        )
    )

    # Composite factor: quality+value tilt
    composite = CompositeFactor(
        name="quality_value_combo",
        components={"quality": 0.6, "value": 0.4},
        source="factors",
        standardize=True,
    )
    model.register_factor(composite)

    custom_factors_df = model.compute_all(universe, base_factors_df=base_factors_df, id_col="ticker")
    print("\nCustom factors head:")
    print(custom_factors_df.head())

    merged = model.merge_with_base_factors(base_factors_df, custom_factors_df)
    print("\nMerged factors head:")
    print(merged.head())
