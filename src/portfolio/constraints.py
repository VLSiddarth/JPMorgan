"""
Portfolio Constraints

Reusable, composable constraint objects for portfolio optimization
and rebalancing, including:

- Per-asset weight bounds
- Sector / group exposure limits
- Turnover constraints
- Concentration constraints

These are designed to be:

- Configurable (dataclasses with type hints)
- Independent of any specific optimizer
- Usable both in optimization and rebalancing validation

Typical usage:

    from src.portfolio.constraints import (
        PortfolioConstraintSet,
        WeightBounds,
        SectorExposureConstraint,
        TurnoverConstraint,
    )

    constraints = PortfolioConstraintSet(
        weight_bounds=WeightBounds(
            min_weight=-0.05,
            max_weight=0.10,
            per_asset_max={"SXXP": 0.15},
        ),
        sector_constraints=[
            SectorExposureConstraint(
                sector="Financials",
                min_weight=0.05,
                max_weight=0.30,
            )
        ],
        turnover_constraint=TurnoverConstraint(max_turnover=0.30),
    )

    ok, violations = constraints.validate(
        current_weights=current_w,
        target_weights=target_w,
        asset_sectors=sector_map,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------


@dataclass
class WeightBounds:
    """
    Global and per-asset weight bounds.

    Attributes:
        min_weight: Global minimum weight (e.g. -0.1 for -10% short allowed).
        max_weight: Global maximum weight (e.g. 0.1 for 10% cap).
        per_asset_min: Optional dict overriding min_weight for specific assets.
        per_asset_max: Optional dict overriding max_weight for specific assets.
    """

    min_weight: float = 0.0
    max_weight: float = 0.10
    per_asset_min: Dict[str, float] = field(default_factory=dict)
    per_asset_max: Dict[str, float] = field(default_factory=dict)

    def validate(self, weights: Mapping[str, float]) -> List[str]:
        """
        Validate that all asset weights are within bounds.

        Args:
            weights: Dict symbol -> target weight.

        Returns:
            List of violation messages (empty if valid).
        """
        violations: List[str] = []

        for symbol, w in weights.items():
            min_w = self.per_asset_min.get(symbol, self.min_weight)
            max_w = self.per_asset_max.get(symbol, self.max_weight)

            if w < min_w - 1e-8:
                msg = f"Weight for {symbol} = {w:.4f} < min {min_w:.4f}"
                violations.append(msg)
                logger.debug(msg)

            if w > max_w + 1e-8:
                msg = f"Weight for {symbol} = {w:.4f} > max {max_w:.4f}"
                violations.append(msg)
                logger.debug(msg)

        return violations


@dataclass
class SectorExposureConstraint:
    """
    Sector or group exposure constraint.

    Attributes:
        sector: Sector / group name this constraint applies to.
        min_weight: Minimum total weight for this sector (optional).
        max_weight: Maximum total weight for this sector (optional).

    Example:
        SectorExposureConstraint(
            sector="Financials",
            min_weight=0.05,
            max_weight=0.25,
        )
    """

    sector: str
    min_weight: Optional[float] = None
    max_weight: Optional[float] = None

    def validate(
        self,
        weights: Mapping[str, float],
        asset_sectors: Mapping[str, str],
    ) -> List[str]:
        """
        Validate sector exposure for this constraint.

        Args:
            weights: Dict symbol -> target weight.
            asset_sectors: Dict symbol -> sector name.

        Returns:
            List of violation messages (empty if valid).
        """
        sector_weight = 0.0
        for symbol, w in weights.items():
            if asset_sectors.get(symbol) == self.sector:
                sector_weight += w

        violations: List[str] = []

        if self.min_weight is not None and sector_weight < self.min_weight - 1e-8:
            msg = (
                f"Sector '{self.sector}' weight {sector_weight:.4f} < "
                f"min {self.min_weight:.4f}"
            )
            violations.append(msg)
            logger.debug(msg)

        if self.max_weight is not None and sector_weight > self.max_weight + 1e-8:
            msg = (
                f"Sector '{self.sector}' weight {sector_weight:.4f} > "
                f"max {self.max_weight:.4f}"
            )
            violations.append(msg)
            logger.debug(msg)

        return violations


@dataclass
class ConcentrationConstraint:
    """
    Portfolio concentration constraint.

    Examples:
        - Max single-name weight
        - Max top-N aggregate weight

    Attributes:
        max_single_weight: Maximum single asset weight allowed.
        max_top_n_weight: Maximum combined weight of top N positions.
        top_n: N for top-N concentration check.
    """

    max_single_weight: Optional[float] = 0.10
    max_top_n_weight: Optional[float] = 0.40
    top_n: int = 10

    def validate(self, weights: Mapping[str, float]) -> List[str]:
        """
        Validate concentration constraints.

        Args:
            weights: Dict symbol -> target weight.

        Returns:
            List of violation messages (empty if valid).
        """
        if not weights:
            return []

        violations: List[str] = []
        w_series = pd.Series(weights).fillna(0.0).abs()

        # Single-name max
        if self.max_single_weight is not None:
            max_w = w_series.max()
            if max_w > self.max_single_weight + 1e-8:
                symbol = w_series.idxmax()
                msg = (
                    f"Max single-name weight {max_w:.4f} in {symbol} "
                    f"> limit {self.max_single_weight:.4f}"
                )
                violations.append(msg)
                logger.debug(msg)

        # Top-N concentration
        if self.max_top_n_weight is not None and self.top_n > 0:
            top_sum = w_series.sort_values(ascending=False).head(self.top_n).sum()
            if top_sum > self.max_top_n_weight + 1e-8:
                msg = (
                    f"Top-{self.top_n} weight {top_sum:.4f} > "
                    f"limit {self.max_top_n_weight:.4f}"
                )
                violations.append(msg)
                logger.debug(msg)

        return violations


@dataclass
class TurnoverConstraint:
    """
    Turnover constraint between current and target portfolio.

    Turnover is defined as:

        0.5 * sum_i |w_target_i - w_current_i|

    which is the standard "one-way" turnover measure.

    Attributes:
        max_turnover: Maximum allowed turnover (e.g. 0.3 for 30%).
    """

    max_turnover: float = 0.30

    def validate(
        self,
        current_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
    ) -> List[str]:
        """
        Validate turnover constraint.

        Args:
            current_weights: Dict symbol -> current weight.
            target_weights: Dict symbol -> target weight.

        Returns:
            List of violation messages (empty if valid).
        """
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        diffs = []
        for sym in all_symbols:
            w0 = float(current_weights.get(sym, 0.0))
            w1 = float(target_weights.get(sym, 0.0))
            diffs.append(abs(w1 - w0))

        turnover = 0.5 * float(np.sum(diffs))
        violations: List[str] = []

        if turnover > self.max_turnover + 1e-6:
            msg = (
                f"Turnover {turnover:.4f} exceeds max {self.max_turnover:.4f}. "
                "Consider scaling down trades or relaxing constraints."
            )
            violations.append(msg)
            logger.debug(msg)

        return violations

    def compute_turnover(
        self,
        current_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
    ) -> float:
        """Return computed turnover."""
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        diffs = [
            abs(float(target_weights.get(sym, 0.0)) - float(current_weights.get(sym, 0.0)))
            for sym in all_symbols
        ]
        return 0.5 * float(np.sum(diffs))


# ---------------------------------------------------------------------------
# Portfolio-wide constraint set
# ---------------------------------------------------------------------------


@dataclass
class PortfolioConstraintSet:
    """
    Container for all portfolio-level constraints.

    This class centralizes validation so both the optimizer and rebalancer
    can run the same checks.

    Attributes:
        weight_bounds: Global/per-asset min/max.
        sector_constraints: List of sector exposure rules.
        concentration_constraint: Single/top-N exposure rules.
        turnover_constraint: Optional turnover limit.
    """

    weight_bounds: Optional[WeightBounds] = None
    sector_constraints: List[SectorExposureConstraint] = field(default_factory=list)
    concentration_constraint: Optional[ConcentrationConstraint] = None
    turnover_constraint: Optional[TurnoverConstraint] = None

    def validate(
        self,
        current_weights: Optional[Mapping[str, float]] = None,
        target_weights: Optional[Mapping[str, float]] = None,
        asset_sectors: Optional[Mapping[str, str]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validate all configured constraints.

        Args:
            current_weights: Dict symbol -> current weight. Required if
                turnover_constraint is set.
            target_weights: Dict symbol -> target weight. Required for
                all constraints.
            asset_sectors: Dict symbol -> sector name. Required if
                sector_constraints are configured.

        Returns:
            (is_valid, violations) where:
                - is_valid: True if no violations.
                - violations: List of human-readable messages.
        """
        if target_weights is None:
            raise ValueError("target_weights is required for constraint validation.")

        violations: List[str] = []

        # Weight bounds
        if self.weight_bounds is not None:
            violations.extend(self.weight_bounds.validate(target_weights))

        # Sector constraints
        if self.sector_constraints:
            if asset_sectors is None:
                logger.warning("Sector constraints set but asset_sectors not provided.")
            else:
                for sc in self.sector_constraints:
                    violations.extend(sc.validate(target_weights, asset_sectors))

        # Concentration
        if self.concentration_constraint is not None:
            violations.extend(self.concentration_constraint.validate(target_weights))

        # Turnover
        if self.turnover_constraint is not None:
            if current_weights is None:
                logger.warning(
                    "TurnoverConstraint configured but current_weights not provided. "
                    "Skipping turnover validation."
                )
            else:
                violations.extend(
                    self.turnover_constraint.validate(current_weights, target_weights)
                )

        is_valid = len(violations) == 0
        if is_valid:
            logger.info("All portfolio constraints satisfied.")
        else:
            logger.warning("Constraint violations detected: %d issue(s).", len(violations))

        return is_valid, violations
