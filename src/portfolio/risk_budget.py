"""
Risk Budgeting Engine

Provides institutional-grade risk budgeting utilities for portfolio construction:
- Equal Risk Contribution (ERC) weights
- Custom risk budgets by asset
- Sector-level risk budgets (map sector targets to asset-level weights)
- Risk contribution diagnostics

Intended usage inside the JPMorganChase project:

    from src.portfolio.risk_budget import RiskBudgetEngine, RiskBudgetConfig

    engine = RiskBudgetEngine()
    weights_erc = engine.equal_risk_contribution(cov_matrix)

    weights_custom = engine.allocate_by_risk_budget(
        cov_matrix=cov_matrix,
        risk_budgets={"SXXP": 0.4, "SX7E": 0.3, "SX8E": 0.3},
    )

    # Sector-level risk budget
    weights_sector = engine.allocate_by_sector_budget(
        cov_matrix=cov_matrix,
        asset_sectors={"SXXP": "Core", "SX7E": "Financials", "SX8E": "Industrials"},
        sector_budgets={"Core": 0.4, "Financials": 0.3, "Industrials": 0.3},
    )

All methods are:
- Type-hinted
- Logging-enabled
- Robust to small numerical issues
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RiskBudgetConfig:
    """
    Configuration for risk budgeting.

    Attributes:
        long_only: If True, constrain weights >= 0.
        weight_sum: Target sum of weights (usually 1.0).
        max_iter: Maximum optimization iterations.
        tol: Tolerance for convergence.
    """

    long_only: bool = True
    weight_sum: float = 1.0
    max_iter: int = 500
    tol: float = 1e-7


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------


class RiskBudgetEngine:
    """
    Risk budgeting engine providing:

    - Equal Risk Contribution (ERC)
    - Custom risk budgets by asset
    - Sector-level risk budgets

    All APIs work with a covariance matrix (numpy array or pandas DataFrame).
    """

    def __init__(self, config: Optional[RiskBudgetConfig] = None) -> None:
        self.config = config or RiskBudgetConfig()
        logger.info(
            "RiskBudgetEngine initialized (long_only=%s, weight_sum=%.2f)",
            self.config.long_only,
            self.config.weight_sum,
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def equal_risk_contribution(
        self,
        cov_matrix: pd.DataFrame | np.ndarray,
        asset_names: Optional[Iterable[str]] = None,
    ) -> pd.Series:
        """
        Compute Equal Risk Contribution (ERC) weights.

        Each asset contributes the same fraction to total portfolio risk.

        Args:
            cov_matrix: Covariance matrix of asset returns (n x n).
                        Can be pandas.DataFrame (with index as asset names)
                        or numpy.ndarray.
            asset_names: Optional list of asset names if cov_matrix is ndarray.

        Returns:
            pd.Series of ERC weights, indexed by asset name.
        """
        cov, names = self._prepare_cov(cov_matrix, asset_names)

        n = cov.shape[0]
        # Initial guess: equal weights
        w0 = np.full(n, 1.0 / n)

        # Target risk contributions: equal
        target_rc = np.full(n, 1.0 / n)

        logger.info("Computing Equal Risk Contribution weights for %d assets.", n)
        w_opt = self._solve_risk_budget(cov, target_rc, w0)

        weights = pd.Series(w_opt, index=names)
        logger.debug("ERC weights: %s", weights.to_dict())

        return weights

    def allocate_by_risk_budget(
        self,
        cov_matrix: pd.DataFrame | np.ndarray,
        risk_budgets: Mapping[str, float],
        asset_names: Optional[Iterable[str]] = None,
    ) -> pd.Series:
        """
        Allocate weights so that each asset contributes a specific fraction
        of total risk (generalized risk-parity).

        Args:
            cov_matrix: Covariance matrix (n x n).
            risk_budgets: Dict asset -> desired risk fraction (should sum to 1).
            asset_names: Optional asset names if cov_matrix is ndarray.

        Returns:
            pd.Series of weights that approximate the desired risk budget.
        """
        cov, names = self._prepare_cov(cov_matrix, asset_names)
        n = cov.shape[0]

        # Normalize provided risk budgets to sum to 1
        rb = pd.Series(risk_budgets).reindex(names).fillna(0.0)
        total_rb = float(rb.sum())
        if total_rb <= 0:
            raise ValueError("risk_budgets must sum to a positive value.")
        rb = rb / total_rb

        target_rc = rb.values
        w0 = np.full(n, 1.0 / n)

        logger.info("Computing custom risk-budget weights for %d assets.", n)
        w_opt = self._solve_risk_budget(cov, target_rc, w0)

        weights = pd.Series(w_opt, index=names)
        logger.debug("Risk budget weights: %s", weights.to_dict())

        return weights

    def allocate_by_sector_budget(
        self,
        cov_matrix: pd.DataFrame | np.ndarray,
        asset_sectors: Mapping[str, str],
        sector_budgets: Mapping[str, float],
        asset_names: Optional[Iterable[str]] = None,
    ) -> pd.Series:
        """
        Allocate weights such that sector-level risk contributions follow
        given sector risk budgets, and within each sector, risk is shared equally.

        High-level process:
            1. Map assets -> sectors
            2. Solve for sector weights according to sector_budgets
               (via simple proportional mapping, not optimization)
            3. Within each sector, compute ERC weights using sub-cov matrix
            4. Combine into asset-level weights

        Args:
            cov_matrix: Covariance matrix (n x n).
            asset_sectors: Dict asset -> sector name.
            sector_budgets: Dict sector -> desired risk share (sum ≈ 1).
            asset_names: Optional asset names if cov_matrix is ndarray.

        Returns:
            pd.Series of asset weights consistent with sector risk budgets.
        """
        cov, names = self._prepare_cov(cov_matrix, asset_names)
        n = cov.shape[0]
        logger.info("Computing sector-level risk budget for %d assets.", n)

        # Normalize sector budgets
        sector_rb = pd.Series(sector_budgets)
        if sector_rb.sum() <= 0:
            raise ValueError("sector_budgets must sum to a positive value.")
        sector_rb = sector_rb / float(sector_rb.sum())

        # Map assets -> sectors
        asset_sector_series = pd.Series(asset_sectors).reindex(names)
        if asset_sector_series.isna().any():
            missing = asset_sector_series[asset_sector_series.isna()].index.tolist()
            logger.warning("Some assets missing sector mapping: %s", missing)

        # Determine sector total weights simply proportional to risk budgets
        # Here we do a simple mapping: sector_weight = risk_budget (already sums to 1)
        sector_weights = sector_rb.copy()

        # Within each sector, use ERC on that sub-covariance
        asset_weights = pd.Series(0.0, index=names)

        for sector, sector_w in sector_weights.items():
            sector_assets = asset_sector_series[asset_sector_series == sector].index
            if len(sector_assets) == 0:
                logger.warning("No assets found for sector '%s'; skipping.", sector)
                continue

            sub_cov = cov.loc[sector_assets, sector_assets]
            logger.info(
                "Computing ERC within sector '%s' for %d assets (sector weight=%.3f).",
                sector,
                len(sector_assets),
                sector_w,
            )
            # Sector-level ERC (weights within sector sum to 1)
            sub_weights_erc = self.equal_risk_contribution(sub_cov, asset_names=sector_assets)

            # Scale by sector weight
            asset_weights.loc[sector_assets] = sector_w * sub_weights_erc

        # Soft renormalization
        total_w = float(asset_weights.sum())
        if total_w > 0:
            asset_weights = asset_weights / total_w * self.config.weight_sum

        logger.debug("Sector-based risk budget weights: %s", asset_weights.to_dict())
        return asset_weights

    def compute_risk_contributions(
        self,
        cov_matrix: pd.DataFrame | np.ndarray,
        weights: Mapping[str, float] | np.ndarray,
        asset_names: Optional[Iterable[str]] = None,
    ) -> Tuple[pd.Series, float]:
        """
        Compute asset-level risk contributions:

            RC_i = w_i * (Σ w)_i / portfolio_vol

        such that sum(RC_i) = portfolio_vol.

        Args:
            cov_matrix: Covariance matrix.
            weights: Dict symbol -> weight OR numpy array.
            asset_names: Asset names if using numpy arrays.

        Returns:
            (rc_series, portfolio_vol) where:
                - rc_series: pd.Series of risk contribution per asset
                - portfolio_vol: total portfolio volatility
        """
        cov, names = self._prepare_cov(cov_matrix, asset_names)

        if isinstance(weights, dict):
            w = pd.Series(weights).reindex(names).fillna(0.0).values
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape[0] != cov.shape[0]:
                raise ValueError("weights length does not match cov_matrix dimension.")

        # Portfolio variance and vol
        port_var = float(w.T @ cov.values @ w)
        port_vol = np.sqrt(max(port_var, 0.0))

        if port_vol == 0.0:
            rc = pd.Series(0.0, index=names)
            return rc, port_vol

        # Marginal contribution: (Σ w)
        mrc = cov.values @ w  # shape (n,)
        # Total risk contribution: w_i * MRC_i / vol
        rc_vals = w * mrc / port_vol

        rc = pd.Series(rc_vals, index=names)
        return rc, port_vol

    # ---------------------------------------------------------------------
    # Internal optimization helpers
    # ---------------------------------------------------------------------

    def _solve_risk_budget(
        self,
        cov: pd.DataFrame,
        target_rc: np.ndarray,
        w0: np.ndarray,
    ) -> np.ndarray:
        """
        Solve risk budget optimization:

            minimize   f(w) = sum_i (RC_i / sum(RC) - b_i)^2
            subject to sum(w_i) = weight_sum
                      (if long_only) w_i >= 0

        where RC_i are risk contributions and b_i are target risk budgets.

        Args:
            cov: Covariance matrix as DataFrame.
            target_rc: Target risk fractions (sum ≈ 1).
            w0: Initial guess for weights.

        Returns:
            Optimal weights as numpy array.
        """
        n = cov.shape[0]
        cov_values = cov.values
        b = target_rc / float(np.sum(target_rc))

        def portfolio_vol(w: np.ndarray) -> float:
            return float(np.sqrt(max(w.T @ cov_values @ w, 0.0)))

        def risk_contributions(w: np.ndarray) -> np.ndarray:
            vol = portfolio_vol(w)
            if vol == 0.0:
                return np.zeros_like(w)
            mrc = cov_values @ w
            return w * mrc / vol  # absolute contribution

        def objective(w: np.ndarray) -> float:
            rc = risk_contributions(w)
            total_rc = float(np.sum(rc))
            if total_rc == 0.0:
                # If portfolio has zero risk, penalize heavily to escape this region
                return 1e6
            rc_frac = rc / total_rc
            return float(np.sum((rc_frac - b) ** 2))

        # Constraints: sum(w) = weight_sum
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - self.config.weight_sum}]

        # Bounds: [0, +inf) if long-only, else None
        if self.config.long_only:
            bounds = tuple((0.0, None) for _ in range(n))
        else:
            bounds = None

        result = minimize(
            objective,
            x0=w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": self.config.max_iter, "ftol": self.config.tol, "disp": False},
        )

        if not result.success:
            logger.warning(
                "Risk budget optimization did not fully converge: %s (status=%s)",
                result.message,
                result.status,
            )

        w_opt = np.asarray(result.x, dtype=float)

        # Soft clean-up: force tiny negatives to zero, renormalize.
        if self.config.long_only:
            w_opt[w_opt < 0] = 0.0
        s = float(np.sum(w_opt))
        if s != 0:
            w_opt = w_opt / s * self.config.weight_sum

        return w_opt

    # ---------------------------------------------------------------------
    # Utility
    # ---------------------------------------------------------------------

    @staticmethod
    def _prepare_cov(
        cov_matrix: pd.DataFrame | np.ndarray,
        asset_names: Optional[Iterable[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Index]:
        """
        Normalize covariance input to a DataFrame with proper asset names.
        """
        if isinstance(cov_matrix, pd.DataFrame):
            cov_df = cov_matrix.copy()
            names = cov_df.index
        else:
            arr = np.asarray(cov_matrix, dtype=float)
            if arr.shape[0] != arr.shape[1]:
                raise ValueError("cov_matrix must be square.")
            if asset_names is None:
                names = pd.Index([f"asset_{i}" for i in range(arr.shape[0])])
            else:
                names = pd.Index(list(asset_names))
                if len(names) != arr.shape[0]:
                    raise ValueError(
                        "Length of asset_names does not match cov_matrix dimension."
                    )
            cov_df = pd.DataFrame(arr, index=names, columns=names)

        return cov_df, cov_df.index


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simple test with 3 assets
    np.random.seed(42)
    names = ["SXXP", "SX7E", "SX8E"]

    # Simulate covariance matrix
    A = np.array([[0.04, 0.01, 0.008],
                  [0.01, 0.09, 0.012],
                  [0.008, 0.012, 0.0625]])
    cov_df = pd.DataFrame(A, index=names, columns=names)

    engine = RiskBudgetEngine()

    print("=== Equal Risk Contribution ===")
    w_erc = engine.equal_risk_contribution(cov_df)
    rc_erc, vol_erc = engine.compute_risk_contributions(cov_df, w_erc)
    print("Weights:\n", w_erc)
    print("Risk contributions:\n", rc_erc)
    print("Portfolio vol:", vol_erc)

    print("\n=== Custom Asset Risk Budget ===")
    rb = {"SXXP": 0.5, "SX7E": 0.25, "SX8E": 0.25}
    w_rb = engine.allocate_by_risk_budget(cov_df, rb)
    rc_rb, vol_rb = engine.compute_risk_contributions(cov_df, w_rb)
    print("Weights:\n", w_rb)
    print("Risk contributions:\n", rc_rb / rc_rb.sum())  # show fractions

    print("\n=== Sector Risk Budget ===")
    sectors = {"SXXP": "Core", "SX7E": "Financials", "SX8E": "Financials"}
    sector_budgets = {"Core": 0.4, "Financials": 0.6}
    w_sec = engine.allocate_by_sector_budget(cov_df, sectors, sector_budgets)
    rc_sec, vol_sec = engine.compute_risk_contributions(cov_df, w_sec)
    print("Weights:\n", w_sec)
    print("Risk contributions:\n", rc_sec / rc_sec.sum())
