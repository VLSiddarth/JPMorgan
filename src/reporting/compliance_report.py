"""
Compliance Reporting Module

Generates institutional-style compliance reports for the JPMorgan
European Equity Dashboard, including:

- Position & exposure limits (per-asset, sector, country, issuer)
- Concentration limits (single-name, top-N)
- Risk limits (VaR, volatility, tracking error, drawdown)
- Trade & restricted list checks

Outputs a structured ComplianceReport object that can be:
- Rendered as text / markdown
- Serialized to JSON-compatible dict
- Used to trigger alerts / emails

This module is intentionally decoupled from specific portfolio
implementations – you pass in positions, weights, risk metrics, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------


class BreachSeverity(str, Enum):
    """Compliance breach severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class RuleType(str, Enum):
    """Type/category of compliance rule."""

    POSITION_LIMIT = "POSITION_LIMIT"
    CONCENTRATION = "CONCENTRATION"
    SECTOR_LIMIT = "SECTOR_LIMIT"
    COUNTRY_LIMIT = "COUNTRY_LIMIT"
    RISK_LIMIT = "RISK_LIMIT"
    RESTRICTED_LIST = "RESTRICTED_LIST"
    TRADE_RULE = "TRADE_RULE"
    OTHER = "OTHER"


@dataclass
class ComplianceRule:
    """
    Definition of a compliance rule.

    Attributes:
        rule_id: Unique identifier (e.g. 'POS_MAX_WEIGHT_SINGLE').
        name: Human-readable name.
        description: Detailed explanation.
        rule_type: Category of rule.
        threshold: Numeric threshold (if applicable).
        metadata: Optional extra info (e.g. {'top_n': 10}).
        severity: Default severity if breached.
    """

    rule_id: str
    name: str
    description: str
    rule_type: RuleType
    threshold: Optional[float] = None
    severity: BreachSeverity = BreachSeverity.WARNING
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceBreach:
    """
    Single compliance breach / flag.

    Attributes:
        rule_id: Which rule was breached.
        rule_type: High-level category.
        severity: Severity level.
        message: Human-readable description.
        details: Structured detail (e.g. offending positions).
        timestamp: When the breach was detected.
    """

    rule_id: str
    rule_type: RuleType
    severity: BreachSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplianceReport:
    """
    Full compliance report for a given portfolio / run.

    Attributes:
        as_of: Timestamp of report.
        portfolio_id: Identifier (e.g. 'EU_EQUITY_THESIS').
        breaches: List of ComplianceBreach objects.
        context: Optional metadata (e.g. risk metrics snapshot).
    """

    as_of: datetime
    portfolio_id: str
    breaches: List[ComplianceBreach] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_compliant(self) -> bool:
        """Return True if there are no WARNING/CRITICAL breaches."""
        return not any(
            b.severity in {BreachSeverity.WARNING, BreachSeverity.CRITICAL}
            for b in self.breaches
        )

    def summary_counts(self) -> Dict[str, int]:
        """Return breach counts by severity."""
        counts: Dict[str, int] = {"INFO": 0, "WARNING": 0, "CRITICAL": 0}
        for b in self.breaches:
            counts[b.severity.value] = counts.get(b.severity.value, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dict representation."""
        return {
            "as_of": self.as_of.isoformat(),
            "portfolio_id": self.portfolio_id,
            "is_compliant": self.is_compliant,
            "summary_counts": self.summary_counts(),
            "breaches": [
                {
                    "rule_id": b.rule_id,
                    "rule_type": b.rule_type.value,
                    "severity": b.severity.value,
                    "message": b.message,
                    "details": b.details,
                    "timestamp": b.timestamp.isoformat(),
                }
                for b in self.breaches
            ],
            "context": self._serialize_context(),
        }

    def _serialize_context(self) -> Dict[str, Any]:
        """Helper to make context JSON-safe (best-effort)."""
        out: Dict[str, Any] = {}
        for k, v in self.context.items():
            if isinstance(v, (pd.Series, pd.DataFrame)):
                out[k] = v.to_dict()
            elif isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out

    def to_markdown(self) -> str:
        """Render a human-readable markdown summary."""
        header = f"# Compliance Report – {self.portfolio_id}\n\n"
        header += f"- As of: **{self.as_of.isoformat()}**\n"
        header += (
            f"- Overall Status: **{'COMPLIANT ✅' if self.is_compliant else 'BREACHES ⚠️'}**\n"
        )
        counts = self.summary_counts()
        header += (
            f"- Breaches: CRITICAL={counts['CRITICAL']}, "
            f"WARNING={counts['WARNING']}, INFO={counts['INFO']}\n\n"
        )

        if not self.breaches:
            header += "_No breaches detected._\n"
            return header

        body = "## Breach Details\n\n"
        for b in self.breaches:
            body += f"### [{b.severity.value}] {b.rule_id} – {b.rule_type.value}\n"
            body += f"- **Time:** {b.timestamp.isoformat()}\n"
            body += f"- **Message:** {b.message}\n"
            if b.details:
                body += f"- **Details:** `{b.details}`\n"
            body += "\n"

        return header + body


# ---------------------------------------------------------------------------
# Reporter Engine
# ---------------------------------------------------------------------------


class ComplianceReporter:
    """
    Generates compliance breaches and aggregate reports.

    This class does not enforce rules; it simply checks and reports.
    Integration with monitoring/alerts is done elsewhere.

    Typical usage:

        reporter = ComplianceReporter(portfolio_id="EU_EQUITY_THESIS")

        breaches = []
        breaches += reporter.check_position_limits(...)
        breaches += reporter.check_concentration(...)
        breaches += reporter.check_risk_limits(...)

        report = reporter.generate_report(
            breaches=breaches,
            context={"risk_metrics": risk_dict, "weights": weights_dict},
        )
    """

    def __init__(self, portfolio_id: str) -> None:
        self.portfolio_id = portfolio_id
        logger.info("ComplianceReporter initialized for portfolio '%s'.", portfolio_id)

    # ------------------------------------------------------------------
    # Position & Exposure Checks
    # ------------------------------------------------------------------

    def check_position_limits(
        self,
        weights: Mapping[str, float],
        max_single_weight: float,
        min_weight: float = -0.05,
        restricted_symbols: Optional[Iterable[str]] = None,
    ) -> List[ComplianceBreach]:
        """
        Check per-position exposure limits and restricted symbols.

        Args:
            weights: Dict symbol -> portfolio weight (including shorts).
            max_single_weight: Maximum absolute weight per asset (e.g. 0.1).
            min_weight: Minimum allowed weight (for long-only, use 0.0).
            restricted_symbols: Optional list of restricted tickers (no exposure allowed).

        Returns:
            List of ComplianceBreach.
        """
        restricted_set = set(restricted_symbols or [])
        breaches: List[ComplianceBreach] = []

        w_series = pd.Series(weights).fillna(0.0)

        # 1) Single-name abs weight check
        for sym, w in w_series.items():
            if abs(w) > max_single_weight + 1e-8:
                msg = (
                    f"Position in {sym} has weight {w:.4f}, exceeding "
                    f"max_single_weight={max_single_weight:.4f}"
                )
                breaches.append(
                    ComplianceBreach(
                        rule_id="POS_MAX_SINGLE_WEIGHT",
                        rule_type=RuleType.POSITION_LIMIT,
                        severity=BreachSeverity.CRITICAL,
                        message=msg,
                        details={"symbol": sym, "weight": float(w), "limit": max_single_weight},
                    )
                )

            if w < min_weight - 1e-8:
                msg = (
                    f"Position in {sym} has weight {w:.4f}, below "
                    f"min_weight={min_weight:.4f}"
                )
                breaches.append(
                    ComplianceBreach(
                        rule_id="POS_MIN_WEIGHT",
                        rule_type=RuleType.POSITION_LIMIT,
                        severity=BreachSeverity.WARNING,
                        message=msg,
                        details={"symbol": sym, "weight": float(w), "limit": min_weight},
                    )
                )

        # 2) Restricted list (no exposure allowed)
        for sym in restricted_set:
            w = float(w_series.get(sym, 0.0))
            if abs(w) > 1e-6:
                msg = f"Restricted security {sym} held with weight {w:.4f}."
                breaches.append(
                    ComplianceBreach(
                        rule_id="POS_RESTRICTED_SECURITY",
                        rule_type=RuleType.RESTRICTED_LIST,
                        severity=BreachSeverity.CRITICAL,
                        message=msg,
                        details={"symbol": sym, "weight": w},
                    )
                )

        return breaches

    def check_sector_limits(
        self,
        weights: Mapping[str, float],
        asset_sectors: Mapping[str, str],
        sector_max: Mapping[str, float],
        sector_min: Optional[Mapping[str, float]] = None,
    ) -> List[ComplianceBreach]:
        """
        Check sector exposure limits.

        Args:
            weights: Dict symbol -> weight.
            asset_sectors: Dict symbol -> sector name.
            sector_max: Dict sector -> max weight.
            sector_min: Dict sector -> min weight (optional).

        Returns:
            List of ComplianceBreach.
        """
        w = pd.Series(weights).fillna(0.0)
        sec = pd.Series(asset_sectors)
        sector_min = sector_min or {}

        breaches: List[ComplianceBreach] = []

        # Compute sector weights
        sector_weights: Dict[str, float] = {}
        for sym, wt in w.items():
            s = sec.get(sym)
            if s is None:
                continue
            sector_weights[s] = sector_weights.get(s, 0.0) + wt

        for sector, w_sec in sector_weights.items():
            max_lim = sector_max.get(sector)
            min_lim = sector_min.get(sector)

            if max_lim is not None and w_sec > max_lim + 1e-8:
                msg = (
                    f"Sector '{sector}' weight {w_sec:.4f} exceeds "
                    f"max limit {max_lim:.4f}."
                )
                breaches.append(
                    ComplianceBreach(
                        rule_id="SEC_MAX_WEIGHT",
                        rule_type=RuleType.SECTOR_LIMIT,
                        severity=BreachSeverity.CRITICAL,
                        message=msg,
                        details={"sector": sector, "weight": float(w_sec), "max_limit": max_lim},
                    )
                )

            if min_lim is not None and w_sec < min_lim - 1e-8:
                msg = (
                    f"Sector '{sector}' weight {w_sec:.4f} below "
                    f"min limit {min_lim:.4f}."
                )
                breaches.append(
                    ComplianceBreach(
                        rule_id="SEC_MIN_WEIGHT",
                        rule_type=RuleType.SECTOR_LIMIT,
                        severity=BreachSeverity.WARNING,
                        message=msg,
                        details={"sector": sector, "weight": float(w_sec), "min_limit": min_lim},
                    )
                )

        return breaches

    def check_concentration(
        self,
        weights: Mapping[str, float],
        max_single_weight: Optional[float] = None,
        max_top_n_weight: Optional[float] = None,
        top_n: int = 10,
    ) -> List[ComplianceBreach]:
        """
        Check concentration constraints.

        Args:
            weights: Dict symbol -> weight.
            max_single_weight: Maximum single-asset weight (optional).
            max_top_n_weight: Maximum combined weight of top N positions (optional).
            top_n: N for top-N concentration (default 10).

        Returns:
            List of ComplianceBreach.
        """
        breaches: List[ComplianceBreach] = []
        if not weights:
            return breaches

        w = pd.Series(weights).fillna(0.0).abs()

        if max_single_weight is not None:
            max_w = float(w.max())
            if max_w > max_single_weight + 1e-8:
                sym = w.idxmax()
                msg = (
                    f"Single-name concentration: {sym} has weight {max_w:.4f} "
                    f"> limit {max_single_weight:.4f}."
                )
                breaches.append(
                    ComplianceBreach(
                        rule_id="CONC_MAX_SINGLE",
                        rule_type=RuleType.CONCENTRATION,
                        severity=BreachSeverity.CRITICAL,
                        message=msg,
                        details={
                            "symbol": sym,
                            "weight": max_w,
                            "max_single_weight": max_single_weight,
                        },
                    )
                )

        if max_top_n_weight is not None and top_n > 0:
            top_sum = float(w.sort_values(ascending=False).head(top_n).sum())
            if top_sum > max_top_n_weight + 1e-8:
                msg = (
                    f"Top-{top_n} concentration {top_sum:.4f} "
                    f"> limit {max_top_n_weight:.4f}."
                )
                breaches.append(
                    ComplianceBreach(
                        rule_id="CONC_MAX_TOPN",
                        rule_type=RuleType.CONCENTRATION,
                        severity=BreachSeverity.WARNING,
                        message=msg,
                        details={
                            "top_n": top_n,
                            "top_n_weight": top_sum,
                            "limit": max_top_n_weight,
                        },
                    )
                )

        return breaches

    # ------------------------------------------------------------------
    # Risk Limit Checks
    # ------------------------------------------------------------------

    def check_risk_limits(
        self,
        risk_metrics: Mapping[str, float],
        risk_limits: Mapping[str, float],
    ) -> List[ComplianceBreach]:
        """
        Check risk metrics against configured limits.

        Typical metrics:
            - var_95
            - volatility_annual
            - tracking_error
            - max_drawdown

        Args:
            risk_metrics: Dict metric -> value (e.g. {'var_95': 0.03}).
            risk_limits: Dict metric -> max allowed value.

        Returns:
            List of ComplianceBreach.
        """
        breaches: List[ComplianceBreach] = []

        for metric, limit in risk_limits.items():
            value = float(risk_metrics.get(metric, np.nan))
            if np.isnan(value):
                continue

            if value > limit + 1e-8:
                msg = f"Risk metric '{metric}' = {value:.4f} exceeds limit {limit:.4f}."
                breaches.append(
                    ComplianceBreach(
                        rule_id=f"RISK_{metric.upper()}_LIMIT",
                        rule_type=RuleType.RISK_LIMIT,
                        severity=BreachSeverity.CRITICAL,
                        message=msg,
                        details={"metric": metric, "value": value, "limit": limit},
                    )
                )

        return breaches

    # ------------------------------------------------------------------
    # Trade & Restricted List Checks
    # ------------------------------------------------------------------

    def check_trade_restrictions(
        self,
        trades: Iterable[Mapping[str, Any]],
        restricted_symbols: Iterable[str],
        max_daily_turnover: Optional[float] = None,
        portfolio_value: Optional[float] = None,
    ) -> List[ComplianceBreach]:
        """
        Check trade list against basic restrictions.

        Each trade is a mapping with at least:
            - 'symbol': str
            - 'action': 'BUY' | 'SELL'
            - 'notional': float (if portfolio_value & max_daily_turnover used)

        Args:
            trades: Iterable of trade dicts.
            restricted_symbols: List of restricted symbols.
            max_daily_turnover: Optional limit on total traded notional
                                as fraction of portfolio value (e.g. 0.2).
            portfolio_value: Total portfolio value (needed if turnover limit set).

        Returns:
            List of ComplianceBreach.
        """
        breaches: List[ComplianceBreach] = []
        restricted_set = set(restricted_symbols or [])

        trades_list = list(trades)

        # 1) Restricted list
        for t in trades_list:
            sym = str(t.get("symbol", ""))
            if sym in restricted_set:
                msg = f"Trade in restricted security {sym} detected."
                breaches.append(
                    ComplianceBreach(
                        rule_id="TRADE_RESTRICTED_SECURITY",
                        rule_type=RuleType.RESTRICTED_LIST,
                        severity=BreachSeverity.CRITICAL,
                        message=msg,
                        details={"trade": dict(t)},
                    )
                )

        # 2) Daily turnover limit
        if max_daily_turnover is not None and portfolio_value:
            notional_sum = 0.0
            for t in trades_list:
                notional = float(t.get("notional", 0.0))
                notional_sum += abs(notional)

            turnover = notional_sum / float(portfolio_value) if portfolio_value > 0 else 0.0
            if turnover > max_daily_turnover + 1e-8:
                msg = (
                    f"Daily turnover {turnover:.4f} exceeds limit "
                    f"{max_daily_turnover:.4f}."
                )
                breaches.append(
                    ComplianceBreach(
                        rule_id="TRADE_MAX_DAILY_TURNOVER",
                        rule_type=RuleType.TRADE_RULE,
                        severity=BreachSeverity.WARNING,
                        message=msg,
                        details={
                            "total_notional": notional_sum,
                            "portfolio_value": portfolio_value,
                            "turnover": turnover,
                            "limit": max_daily_turnover,
                        },
                    )
                )

        return breaches

    # ------------------------------------------------------------------
    # Aggregate report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        breaches: List[ComplianceBreach],
        context: Optional[Dict[str, Any]] = None,
    ) -> ComplianceReport:
        """
        Aggregate breaches into a ComplianceReport.

        Args:
            breaches: All breaches collected in this run.
            context: Optional context (weights, risk_metrics, etc.)

        Returns:
            ComplianceReport instance.
        """
        ctx = context or {}
        report = ComplianceReport(
            as_of=datetime.utcnow(),
            portfolio_id=self.portfolio_id,
            breaches=breaches,
            context=ctx,
        )

        counts = report.summary_counts()
        logger.info(
            "Generated compliance report for '%s': CRITICAL=%d, WARNING=%d, INFO=%d",
            self.portfolio_id,
            counts.get("CRITICAL", 0),
            counts.get("WARNING", 0),
            counts.get("INFO", 0),
        )

        return report


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    reporter = ComplianceReporter(portfolio_id="EU_EQUITY_THESIS")

    # Fake weights
    weights = {"SXXP": 0.35, "SX7E": 0.25, "SX8E": 0.22, "SXS8": 0.18}
    sectors = {"SXXP": "Core", "SX7E": "Financials", "SX8E": "Industrials", "SXS8": "Tech"}

    breaches: List[ComplianceBreach] = []
    breaches += reporter.check_position_limits(weights, max_single_weight=0.30)
    breaches += reporter.check_concentration(
        weights, max_single_weight=0.30, max_top_n_weight=0.75, top_n=3
    )
    breaches += reporter.check_sector_limits(
        weights,
        asset_sectors=sectors,
        sector_max={"Financials": 0.30, "Tech": 0.15},
    )
    breaches += reporter.check_risk_limits(
        risk_metrics={"var_95": 0.035, "volatility_annual": 0.18},
        risk_limits={"var_95": 0.03, "volatility_annual": 0.20},
    )
    breaches += reporter.check_trade_restrictions(
        trades=[
            {"symbol": "SAN", "action": "BUY", "notional": 500_000},
            {"symbol": "SXXP", "action": "SELL", "notional": 250_000},
        ],
        restricted_symbols=["SAN"],
        max_daily_turnover=0.30,
        portfolio_value=2_000_000,
    )

    report = reporter.generate_report(
        breaches,
        context={"weights": weights},
    )

    print(report.to_markdown())
