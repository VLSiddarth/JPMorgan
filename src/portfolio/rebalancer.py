"""
Portfolio Rebalancer

Takes current holdings and target weights, and produces:

- Trade list (BUY / SELL with quantities & notionals)
- Turnover statistics
- Post-trade weights (approximate)
- Constraint validation (via PortfolioConstraintSet if provided)

This is designed to sit on top of:
- src/portfolio/optimizer.py (which gives you target weights)
- src/portfolio/constraints.py (to validate turnover, caps, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .constraints import PortfolioConstraintSet, TurnoverConstraint

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """
    Representation of a single trade instruction.

    Attributes:
        symbol: Asset identifier (ticker).
        action: 'BUY' or 'SELL'.
        quantity: Number of units to trade (can be float, rounding applied later).
        price: Last price used for notional computation.
        notional: Trade value (quantity * price), signed (+ buy, - sell).
        weight_before: Pre-trade portfolio weight (approx).
        weight_after: Post-trade portfolio weight (target, approx).
    """

    symbol: str
    action: str
    quantity: float
    price: float
    notional: float
    weight_before: float
    weight_after: float


@dataclass
class RebalanceResult:
    """
    Full result of a rebalance calculation.

    Attributes:
        trades: List of Trade objects.
        turnover: One-way turnover (0..1).
        target_weights: Final target weights used.
        constraint_violations: Any constraint violations detected.
    """

    trades: List[Trade]
    turnover: float
    target_weights: Dict[str, float]
    constraint_violations: List[str] = field(default_factory=list)


@dataclass
class RebalancerConfig:
    """
    Configuration for the rebalancer.

    Attributes:
        min_trade_notional: Minimum absolute notional for a trade (smaller ones dropped).
        rounding_precision: Number of decimals for quantities (e.g. 0 for whole shares).
        cash_symbol: Optional symbol used to represent cash.
        enforce_turnover_limit: If True, scale target weights to respect turnover constraint.
    """

    min_trade_notional: float = 100.0
    rounding_precision: int = 2
    cash_symbol: Optional[str] = "CASH"
    enforce_turnover_limit: bool = True


class PortfolioRebalancer:
    """
    Portfolio rebalancer that:

    - Computes differences between current and target weights
    - Translates those differences into trades, given prices and portfolio value
    - Optionally enforces turnover limits by scaling towards current weights
    - Validates constraints via PortfolioConstraintSet
    """

    def __init__(
        self,
        config: Optional[RebalancerConfig] = None,
        constraints: Optional[PortfolioConstraintSet] = None,
    ) -> None:
        self.config = config or RebalancerConfig()
        self.constraints = constraints

        logger.info(
            "PortfolioRebalancer initialized (min_trade_notional=%.2f, rounding_precision=%d)",
            self.config.min_trade_notional,
            self.config.rounding_precision,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def rebalance(
        self,
        current_positions: Mapping[str, float],
        prices: Mapping[str, float],
        target_weights: Mapping[str, float],
        portfolio_value: float,
    ) -> RebalanceResult:
        """
        Compute trades required to move from current positions to target weights.

        Args:
            current_positions: Dict symbol -> current quantity.
            prices: Dict symbol -> last price.
            target_weights: Dict symbol -> desired portfolio weight (must sum ~1 incl. cash if used).
            portfolio_value: Total portfolio market value (incl. cash if tracked explicitly).

        Returns:
            RebalanceResult with trades, turnover, and constraint violations.
        """
        if portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive.")

        logger.info("Starting rebalance: %d current positions, %d target weights.",
                    len(current_positions), len(target_weights))

        # 1) Compute current weights from positions & prices
        current_weights = self._compute_current_weights(current_positions, prices, portfolio_value)

        # 2) Optionally scale target weights to satisfy turnover constraint
        adjusted_target_weights = dict(target_weights)

        turnover_limit = None
        if self.constraints and self.constraints.turnover_constraint:
            turnover_limit = self.constraints.turnover_constraint.max_turnover

        if self.config.enforce_turnover_limit and turnover_limit is not None:
            adjusted_target_weights = self._scale_to_turnover_limit(
                current_weights=current_weights,
                target_weights=adjusted_target_weights,
                turnover_constraint=self.constraints.turnover_constraint,
            )

        # 3) Generate trades from current vs adjusted target weights
        trades = self._generate_trades(
            current_weights=current_weights,
            target_weights=adjusted_target_weights,
            prices=prices,
            portfolio_value=portfolio_value,
        )

        # 4) Compute final turnover
        turnover = self._compute_turnover(current_weights, adjusted_target_weights)

        # 5) Validate constraints (post-scaling)
        violations: List[str] = []
        if self.constraints is not None:
            ok, violations = self.constraints.validate(
                current_weights=current_weights,
                target_weights=adjusted_target_weights,
                asset_sectors=None,  # can be provided by caller if needed
            )
            if not ok:
                logger.warning("Constraint violations during rebalance: %s", violations)

        return RebalanceResult(
            trades=trades,
            turnover=turnover,
            target_weights=adjusted_target_weights,
            constraint_violations=violations,
        )

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _compute_current_weights(
        self,
        current_positions: Mapping[str, float],
        prices: Mapping[str, float],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Compute current portfolio weights from quantities and prices.

        Args:
            current_positions: Dict symbol -> quantity.
            prices: Dict symbol -> price.
            portfolio_value: Total portfolio value.

        Returns:
            Dict symbol -> weight.
        """
        values: Dict[str, float] = {}
        for sym, qty in current_positions.items():
            price = float(prices.get(sym, 0.0))
            values[sym] = float(qty) * price

        total_value = float(sum(values.values()))
        if total_value <= 0:
            logger.warning("Total marked-to-market value is non-positive; using portfolio_value instead.")
            total_value = portfolio_value

        weights = {sym: val / total_value for sym, val in values.items()}
        logger.debug("Computed current weights for %d symbols.", len(weights))
        return weights

    def _compute_turnover(
        self,
        current_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
    ) -> float:
        """One-way turnover = 0.5 * sum |w1 - w0|."""
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        diffs = [
            abs(float(target_weights.get(sym, 0.0)) - float(current_weights.get(sym, 0.0)))
            for sym in all_symbols
        ]
        return 0.5 * float(np.sum(diffs))

    def _scale_to_turnover_limit(
        self,
        current_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
        turnover_constraint: TurnoverConstraint,
    ) -> Dict[str, float]:
        """
        Scale target weights towards current weights to respect turnover limit.

        We use a simple convex combination:

            w_scaled = w0 + alpha * (w_target - w0)

        and solve for alpha in [0, 1] such that turnover <= max_turnover.

        If current turnover already <= limit, alpha = 1 (no change).
        """
        max_t = turnover_constraint.max_turnover
        current_t = turnover_constraint.compute_turnover(current_weights, target_weights)

        if current_t <= max_t + 1e-8:
            logger.info("Turnover %.4f within limit %.4f. No scaling needed.", current_t, max_t)
            return dict(target_weights)

        # Find alpha in [0, 1] by simple bisection
        logger.info("Turnover %.4f exceeds limit %.4f. Scaling towards current.", current_t, max_t)

        def turnover_for_alpha(alpha: float) -> float:
            scaled: Dict[str, float] = {}
            all_symbols = set(current_weights.keys()) | set(target_weights.keys())
            for sym in all_symbols:
                w0 = float(current_weights.get(sym, 0.0))
                wt = float(target_weights.get(sym, 0.0))
                scaled[sym] = w0 + alpha * (wt - w0)
            return turnover_constraint.compute_turnover(current_weights, scaled)

        low, high = 0.0, 1.0
        best_alpha = 0.0

        for _ in range(40):  # enough iterations for convergence
            mid = 0.5 * (low + high)
            t_mid = turnover_for_alpha(mid)
            if t_mid <= max_t:
                best_alpha = mid
                low = mid
            else:
                high = mid

        logger.info("Scaled target weights with alpha=%.4f to meet turnover constraint.", best_alpha)

        scaled_weights: Dict[str, float] = {}
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        for sym in all_symbols:
            w0 = float(current_weights.get(sym, 0.0))
            wt = float(target_weights.get(sym, 0.0))
            scaled_weights[sym] = w0 + best_alpha * (wt - w0)

        # Optionally renormalize to sum close to 1 (excluding explicit cash if configured)
        scaled_weights = self._renormalize_weights(scaled_weights)

        return scaled_weights

    def _renormalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Renormalize weights so that:
          - Sum of non-cash assets ≈ 1 (if cash_symbol is not used), OR
          - Sum of all assets + cash ≈ 1 (if cash_symbol is used).

        This is a soft normalization for robustness.
        """
        cash_sym = self.config.cash_symbol
        w = pd.Series(weights).fillna(0.0)

        if cash_sym and cash_sym in w.index:
            non_cash = w.drop(cash_sym)
            s = float(non_cash.sum())
            if s != 0:
                scale = 1.0 - float(w[cash_sym])
                non_cash_scaled = non_cash / s * scale
                w.update(non_cash_scaled)
        else:
            s = float(w.sum())
            if s != 0:
                w = w / s

        return {sym: float(val) for sym, val in w.items()}

    def _generate_trades(
        self,
        current_weights: Mapping[str, float],
        target_weights: Mapping[str, float],
        prices: Mapping[str, float],
        portfolio_value: float,
    ) -> List[Trade]:
        """
        Convert weight differences into trades.

        Args:
            current_weights: Dict symbol -> current weight.
            target_weights: Dict symbol -> target weight.
            prices: Dict symbol -> price.
            portfolio_value: Total portfolio value.

        Returns:
            List of Trade objects with min_trade_notional filter applied.
        """
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        trades: List[Trade] = []

        for sym in sorted(all_symbols):
            if self.config.cash_symbol and sym == self.config.cash_symbol:
                # We do not trade the synthetic cash symbol directly here.
                continue

            w0 = float(current_weights.get(sym, 0.0))
            w1 = float(target_weights.get(sym, 0.0))
            delta_w = w1 - w0

            if abs(delta_w) < 1e-6:
                continue  # no trade

            price = float(prices.get(sym, 0.0))
            if price <= 0:
                logger.warning("Missing or non-positive price for %s; skipping trade.", sym)
                continue

            notional = delta_w * portfolio_value
            quantity = notional / price

            # Apply min trade notional filter
            if abs(notional) < self.config.min_trade_notional:
                logger.debug(
                    "Skipping tiny trade in %s (notional=%.2f < %.2f).",
                    sym,
                    notional,
                    self.config.min_trade_notional,
                )
                continue

            # Round quantity
            rounded_qty = round(quantity, self.config.rounding_precision)
            if abs(rounded_qty) < 1e-6:
                continue

            action = "BUY" if rounded_qty > 0 else "SELL"
            final_notional = rounded_qty * price

            trade = Trade(
                symbol=sym,
                action=action,
                quantity=rounded_qty,
                price=price,
                notional=final_notional,
                weight_before=w0,
                weight_after=w1,
            )
            trades.append(trade)

        logger.info("Generated %d trades after filters.", len(trades))
        return trades


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simple self-test
    current_positions = {
        "SXXP": 1000,
        "SX7E": 500,
        "SX8E": 400,
    }
    prices = {
        "SXXP": 450.0,
        "SX7E": 120.0,
        "SX8E": 90.0,
    }
    portfolio_value = sum(q * prices[sym] for sym, q in current_positions.items())

    target_weights = {
        "SXXP": 0.50,
        "SX7E": 0.30,
        "SX8E": 0.20,
    }

    turnover_c = TurnoverConstraint(max_turnover=0.25)
    constraints = PortfolioConstraintSet(turnover_constraint=turnover_c)

    rebalancer = PortfolioRebalancer(constraints=constraints)
    result = rebalancer.rebalance(
        current_positions=current_positions,
        prices=prices,
        target_weights=target_weights,
        portfolio_value=portfolio_value,
    )

    print("\n=== Trades ===")
    for t in result.trades:
        print(f"{t.action:4} {t.symbol:5}  qty={t.quantity:.2f}  notional={t.notional:,.2f}")

    print(f"\nTurnover: {result.turnover:.2%}")
    print("Violations:", result.constraint_violations)
