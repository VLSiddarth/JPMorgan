"""
Signal Generator Orchestrator

High-level signal engine that combines:
- Momentum signals
- Value signals
- Macro / risk signals
- Thesis-specific meta signals (JPM Europe OW thesis)

Outputs:
- A list of structured signal dictionaries for the dashboard
- An overall recommendation (BUY / HOLD / REDUCE) with allocation guidance

This module is designed to be used by the Streamlit app:

    from src.analytics.signals.generator import SignalGenerator

    gen = SignalGenerator()
    signals = gen.generate_signals(data)
    overall = gen.get_overall_recommendation()

    # for CSV export
    df_signals = gen.format_signals_for_display()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Optional imports – engines you already generated earlier
try:
    from .momentum import MomentumSignalEngine
except Exception:  # pragma: no cover - optional dependency
    MomentumSignalEngine = None  # type: ignore

try:
    from .value import ValueSignalEngine
except Exception:  # pragma: no cover
    ValueSignalEngine = None  # type: ignore

try:
    from .macro import MacroSignalEngine
except Exception:  # pragma: no cover
    MacroSignalEngine = None  # type: ignore


# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------

@dataclass
class SignalGeneratorConfig:
    """
    Configuration for signal generation and thresholds.

    If config/thresholds.yml exists, it will override these defaults.
    """

    # Relative performance thresholds (EU vs US 3M %)
    rel_perf_strong_buy: float = -5.0   # EU underperforming by >5% → contrarian BUY
    rel_perf_buy: float = -2.0          # EU underperforming by >2%
    rel_perf_reduce: float = 5.0        # EU outperforming by >5% → take profits

    # Valuation gap P/E: US - EU
    pe_gap_value_buy: float = 8.0       # Gap > 8x → Value BUY
    pe_gap_warning: float = 5.0         # Gap narrowing below 5x

    # FR-DE spread thresholds (bps)
    fr_de_low_risk: float = 40.0
    fr_de_high_risk: float = 80.0

    # Credit impulse (% of GDP)
    credit_impulse_supportive: float = 3.0

    # Internal scoring weights for overall thesis score
    weight_rel_perf: float = 2.0
    weight_eps_growth: float = 2.0
    weight_valuation: float = 1.5
    weight_fragmentation: float = 2.0
    weight_credit_impulse: float = 1.0
    weight_sentiment: float = 1.0  # optional if sentiment is available

    # Sentiment thresholds (-100 to +100)
    sentiment_bullish: float = 30.0
    sentiment_bearish: float = -30.0

    @classmethod
    def from_yaml(cls) -> "SignalGeneratorConfig":
        """
        Load thresholds from config/thresholds.yml if present.
        Otherwise return defaults.
        """
        # Try several likely locations relative to this file
        try:
            this_file = Path(__file__).resolve()
            # project_root = .../JPMorganChase
            project_root = this_file.parents[4]
            cfg_path = project_root / "config" / "thresholds.yml"
        except Exception:
            cfg_path = Path("config/thresholds.yml")

        if not cfg_path.exists():
            logger.info("No thresholds.yml found at %s. Using default SignalGeneratorConfig.", cfg_path)
            return cls()

        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Failed to load thresholds.yml (%s). Using defaults. Error: %s", cfg_path, exc)
            return cls()

        # We only read what we understand; ignore the rest.
        kwargs: Dict[str, Any] = {}

        signals_cfg = raw.get("signals", {})
        rel_cfg = signals_cfg.get("relative_performance", {})
        kwargs["rel_perf_strong_buy"] = rel_cfg.get("strong_buy", cls.rel_perf_strong_buy)
        kwargs["rel_perf_buy"] = rel_cfg.get("buy", cls.rel_perf_buy)
        kwargs["rel_perf_reduce"] = signals_cfg.get("take_profit", {}).get("reduce", cls.rel_perf_reduce)

        frag_cfg = signals_cfg.get("fragmentation_risk", {})
        kwargs["fr_de_high_risk"] = frag_cfg.get("high_risk", cls.fr_de_high_risk)
        kwargs["fr_de_low_risk"] = frag_cfg.get("low_risk", cls.fr_de_low_risk)

        credit_cfg = signals_cfg.get("credit_impulse", {})
        kwargs["credit_impulse_supportive"] = credit_cfg.get("supportive", cls.credit_impulse_supportive)

        sentiment_cfg = signals_cfg.get("sentiment", {})
        kwargs["sentiment_bullish"] = sentiment_cfg.get("bullish", cls.sentiment_bullish)
        kwargs["sentiment_bearish"] = sentiment_cfg.get("bearish", cls.sentiment_bearish)

        logger.info("Loaded SignalGeneratorConfig from %s", cfg_path)
        return cls(**kwargs)


# -----------------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------------

@dataclass
class SignalGenerator:
    """
    High-level signal orchestrator.

    Responsibilities:
      - Call momentum/value/macro engines (if available).
      - Generate thesis-specific top-down signals.
      - Aggregate everything into a clean list for the dashboard.
      - Compute an overall recommendation (BUY/HOLD/REDUCE).

    The instance keeps the last `data` and `signals` internally so that
    `get_overall_recommendation()` and `format_signals_for_display()` work
    without extra arguments (compatible with existing app.py).
    """

    config: SignalGeneratorConfig = field(default_factory=SignalGeneratorConfig.from_yaml)
    momentum_engine: Any = field(default=None, init=False)
    value_engine: Any = field(default=None, init=False)
    macro_engine: Any = field(default=None, init=False)

    _last_data: Optional[Dict[str, Any]] = field(default=None, init=False)
    _last_signals: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # Initialize sub-engines if available
        if MomentumSignalEngine is not None:
            try:
                self.momentum_engine = MomentumSignalEngine()
                logger.info("MomentumSignalEngine initialized.")
            except Exception as exc:
                logger.error("Failed to initialize MomentumSignalEngine: %s", exc, exc_info=True)

        if ValueSignalEngine is not None:
            try:
                self.value_engine = ValueSignalEngine()
                logger.info("ValueSignalEngine initialized.")
            except Exception as exc:
                logger.error("Failed to initialize ValueSignalEngine: %s", exc, exc_info=True)

        if MacroSignalEngine is not None:
            try:
                self.macro_engine = MacroSignalEngine()
                logger.info("MacroSignalEngine initialized.")
            except Exception as exc:
                logger.error("Failed to initialize MacroSignalEngine: %s", exc, exc_info=True)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate_signals(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Main entry point: generate all signals given the dashboard data dict.

        Args:
            data: Dictionary produced by your data pipeline (indices, macro,
                  valuations, sectors, baskets, sentiment, etc.)

        Returns:
            List of signal dicts with fields:
                - id
                - type
                - title
                - message
                - conviction
                - action
                - target_allocation
                - timeframe
                - metrics
                - timestamp (datetime)
        """
        self._last_data = data
        signals: List[Dict[str, Any]] = []

        # 1) Sub-engines (momentum/value/macro factor signals)
        signals.extend(self._run_sub_engines(data))

        # 2) Top-down thesis-specific signals (CIO-level)
        signals.extend(self._generate_thesis_signals(data))

        # De-duplicate by id (keep most recent / last)
        deduped: Dict[str, Dict[str, Any]] = {}
        for s in signals:
            sid = s.get("id") or f"signal_{len(deduped) + 1}"
            s["id"] = sid
            deduped[sid] = s

        self._last_signals = list(deduped.values())
        logger.info("Generated %d aggregated signals.", len(self._last_signals))

        return self._last_signals

    def get_overall_recommendation(self) -> Dict[str, Any]:
        """
        Compute overall recommendation (BUY/HOLD/REDUCE) based on:
        - relative performance EU vs US
        - EPS growth
        - valuation gap
        - fragmentation risk (FR-DE spread)
        - credit impulse
        - sentiment (if available)
        - presence of high-severity risk signals

        Returns:
            dict with keys:
                - recommendation: 'BUY' | 'HOLD' | 'REDUCE'
                - allocation: human-readable guidance
                - score: internal numeric score
                - summary: explanation
        """
        if self._last_data is None:
            return {
                "recommendation": "HOLD",
                "allocation": "Neutral Europe vs benchmark (0% active weight)",
                "score": 0.0,
                "summary": "No data available; defaulting to neutral stance.",
            }

        data = self._last_data
        cfg = self.config

        indices = data.get("indices", {})
        valuations = data.get("valuations", {})
        macro = data.get("macro", {})
        sentiment_block = data.get("sentiment", {})

        # Core metrics
        rel_perf = float(indices.get("relative_performance", 0.0))  # 3M EU vs US
        eps_growth = float(data.get("eps_growth_2026", 12.0))
        pe_gap = float(valuations.get("pe_gap", 7.0))  # US - EU P/E
        fr_de_spread = float(macro.get("fr_de_spread", 70.0))  # bps
        credit_impulse = float(data.get("credit_impulse", 3.0))

        # Sentiment score (-100..+100) if present
        sentiment_score = None
        if isinstance(sentiment_block, dict):
            sentiment_score = sentiment_block.get("score")

        # Start scoring
        score = 0.0
        details: List[str] = []

        # Relative performance – contrarian
        if rel_perf <= cfg.rel_perf_strong_buy:
            score += cfg.weight_rel_perf * 1.5
            details.append(f"Europe underperformed US by {rel_perf:.1f}% (contrarian positive).")
        elif rel_perf <= cfg.rel_perf_buy:
            score += cfg.weight_rel_perf
            details.append(f"Europe modestly underperformed US by {rel_perf:.1f}% (supportive).")
        elif rel_perf >= cfg.rel_perf_reduce:
            score -= cfg.weight_rel_perf
            details.append(f"Europe outperformed US by {rel_perf:.1f}% (take-profit risk).")
        else:
            details.append(f"Europe performance vs US is neutral at {rel_perf:.1f}%.")

        # EPS growth
        if eps_growth >= 12.0:
            score += cfg.weight_eps_growth
            details.append(f"2026 EPS growth at {eps_growth:.1f}% (supports OW).")
        elif eps_growth >= 8.0:
            score += cfg.weight_eps_growth * 0.5
            details.append(f"EPS growth at {eps_growth:.1f}% (moderately supportive).")
        else:
            score -= cfg.weight_eps_growth * 0.5
            details.append(f"EPS growth at {eps_growth:.1f}% (below thesis).")

        # Valuation gap (US - EU P/E) – wider gap is good for value
        if pe_gap >= cfg.pe_gap_value_buy:
            score += cfg.weight_valuation
            details.append(f"EU trades at {pe_gap:.1f}x discount to US (strong value support).")
        elif pe_gap <= cfg.pe_gap_warning:
            score -= cfg.weight_valuation * 0.5
            details.append(f"Valuation gap narrowed to {pe_gap:.1f}x (less cushion).")
        else:
            details.append(f"Valuation gap at {pe_gap:.1f}x (neutral).")

        # Fragmentation risk (FR-DE spread)
        if fr_de_spread < cfg.fr_de_low_risk:
            score += cfg.weight_fragmentation * 0.5
            details.append(f"FR-DE spread at {fr_de_spread:.0f} bps (low fragmentation risk).")
        elif fr_de_spread <= cfg.fr_de_high_risk:
            details.append(f"FR-DE spread at {fr_de_spread:.0f} bps (monitor fragmentation).")
        else:
            score -= cfg.weight_fragmentation
            details.append(f"FR-DE spread at {fr_de_spread:.0f} bps (high fragmentation risk).")

        # Credit impulse
        if credit_impulse >= cfg.credit_impulse_supportive:
            score += cfg.weight_credit_impulse
            details.append(f"EZ credit impulse at {credit_impulse:.1f}% (supports risk assets).")
        else:
            details.append(f"EZ credit impulse at {credit_impulse:.1f}% (not strongly supportive).")

        # Sentiment (if available)
        if sentiment_score is not None:
            try:
                s = float(sentiment_score)
                if s >= cfg.sentiment_bullish:
                    score += cfg.weight_sentiment * 0.5
                    details.append(f"News sentiment is bullish at {s:.1f}.")
                elif s <= cfg.sentiment_bearish:
                    score -= cfg.weight_sentiment * 0.5
                    details.append(f"News sentiment is bearish at {s:.1f}.")
                else:
                    details.append(f"News sentiment is neutral at {s:.1f}.")
            except (TypeError, ValueError):
                pass

        # Penalize if any high-severity risk alerts
        if any(self._is_high_severity_risk(s) for s in self._last_signals):
            score -= 1.0
            details.append("High-severity risk alerts active (penalizing score).")

        # Map score to recommendation
        recommendation: str
        allocation: str

        if score >= 5.0:
            recommendation = "BUY"
            allocation = "Overweight Europe by +5–10% vs benchmark (high conviction)."
        elif score >= 3.0:
            recommendation = "BUY"
            allocation = "Modest overweight Europe by +2–5% vs benchmark."
        elif score >= 1.0:
            recommendation = "HOLD"
            allocation = "Neutral to slight overweight Europe (0–2% active weight)."
        elif score >= -1.0:
            recommendation = "HOLD"
            allocation = "Neutral Europe vs benchmark (0% active weight)."
        else:
            recommendation = "REDUCE"
            allocation = "Underweight Europe by -2–5% vs benchmark."

        summary = " ".join(details)

        result = {
            "recommendation": recommendation,
            "allocation": allocation,
            "score": round(score, 2),
            "summary": summary,
        }

        logger.info("Computed overall recommendation: %s (score=%.2f)", recommendation, score)
        return result

    def format_signals_for_display(self) -> pd.DataFrame:
        """
        Format last generated signals into a DataFrame suitable for:
        - Streamlit tables
        - CSV export

        Returns:
            DataFrame with nice columns:
                ['id', 'timestamp', 'type', 'title', 'conviction',
                 'action', 'timeframe', 'message', 'target_allocation', 'metrics']
        """
        if not self._last_signals:
            return pd.DataFrame(
                columns=[
                    "id",
                    "timestamp",
                    "type",
                    "title",
                    "conviction",
                    "action",
                    "timeframe",
                    "message",
                    "target_allocation",
                    "metrics",
                ]
            )

        rows = []
        for s in self._last_signals:
            rows.append(
                {
                    "id": s.get("id"),
                    "timestamp": s.get("timestamp"),
                    "type": s.get("type"),
                    "title": s.get("title"),
                    "conviction": s.get("conviction"),
                    "action": s.get("action"),
                    "timeframe": s.get("timeframe"),
                    "message": s.get("message"),
                    "target_allocation": s.get("target_allocation"),
                    "metrics": s.get("metrics", {}),
                }
            )

        df = pd.DataFrame(rows)

        # Ensure timestamp is datetime for pretty display
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        return df.sort_values("timestamp", ascending=False, na_position="last")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _run_sub_engines(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call momentum/value/macro engines if available."""
        signals: List[Dict[str, Any]] = []

        if self.momentum_engine is not None:
            try:
                momentum_signals = self.momentum_engine.generate_signals(data)
                signals.extend(momentum_signals)
                logger.info("Momentum engine produced %d signals.", len(momentum_signals))
            except Exception as exc:
                logger.error("Error in MomentumSignalEngine: %s", exc, exc_info=True)

        if self.value_engine is not None:
            try:
                value_signals = self.value_engine.generate_signals(data)
                signals.extend(value_signals)
                logger.info("Value engine produced %d signals.", len(value_signals))
            except Exception as exc:
                logger.error("Error in ValueSignalEngine: %s", exc, exc_info=True)

        if self.macro_engine is not None:
            try:
                macro_signals = self.macro_engine.generate_signals(data)
                signals.extend(macro_signals)
                logger.info("Macro engine produced %d signals.", len(macro_signals))
            except Exception as exc:
                logger.error("Error in MacroSignalEngine: %s", exc, exc_info=True)

        return signals

    def _generate_thesis_signals(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate CIO-level signals tied directly to the JPM Europe OW thesis:
        - Relative performance
        - EPS growth
        - Valuation gap
        - Fragmentation (FR-DE spread)
        - Credit impulse
        """
        cfg = self.config
        indices = data.get("indices", {})
        valuations = data.get("valuations", {})
        macro = data.get("macro", {})

        rel_perf = float(indices.get("relative_performance", 0.0))
        eps_growth = float(data.get("eps_growth_2026", 12.0))
        pe_gap = float(valuations.get("pe_gap", 7.0))
        fr_de_spread = float(macro.get("fr_de_spread", 70.0))
        credit_impulse = float(data.get("credit_impulse", 3.0))

        now = datetime.utcnow()
        signals: List[Dict[str, Any]] = []

        # Relative performance signal
        if rel_perf <= cfg.rel_perf_strong_buy:
            signals.append(
                self._build_signal(
                    sid="rel_perf_strong_buy",
                    s_type="STRONG_BUY",
                    title="Europe Deep Underperformance vs US",
                    message=f"Europe has underperformed the US by {rel_perf:.1f}% over the last 3 months. "
                            f"Historically this creates attractive entry points.",
                    conviction="HIGH",
                    action="Increase Europe exposure aggressively on a 6–12 month horizon.",
                    target_allocation="Overweight EU equities by +5–10% vs benchmark.",
                    timeframe="6–12 months",
                    metrics={"relative_performance_3m": rel_perf},
                    timestamp=now,
                )
            )
        elif rel_perf <= cfg.rel_perf_buy:
            signals.append(
                self._build_signal(
                    sid="rel_perf_buy",
                    s_type="BUY",
                    title="Europe Undervalued vs US (Performance Gap)",
                    message=f"Europe has underperformed the US by {rel_perf:.1f}% over the last 3 months, "
                            f"supporting a contrarian OW stance.",
                    conviction="MEDIUM",
                    action="Tilt portfolios moderately towards Europe.",
                    target_allocation="Overweight EU equities by +2–5% vs benchmark.",
                    timeframe="3–9 months",
                    metrics={"relative_performance_3m": rel_perf},
                    timestamp=now,
                )
            )
        elif rel_perf >= cfg.rel_perf_reduce:
            signals.append(
                self._build_signal(
                    sid="rel_perf_reduce",
                    s_type="RISK_ALERT",
                    title="Europe Outperformance – Take Profit Risk",
                    message=f"Europe has outperformed the US by {rel_perf:.1f}% over the last 3 months. "
                            f"Consider partial profit-taking.",
                    conviction="MEDIUM",
                    action="Gradually reduce Europe overweight and lock in gains.",
                    target_allocation="Reduce EU OW back towards neutral.",
                    timeframe="1–3 months",
                    metrics={"relative_performance_3m": rel_perf},
                    timestamp=now,
                )
            )

        # EPS growth signal
        if eps_growth >= 12.0:
            signals.append(
                self._build_signal(
                    sid="eps_growth_supportive",
                    s_type="FUNDAMENTAL_BULLISH",
                    title="Eurozone EPS Growth Supports OW Thesis",
                    message=f"2026 EPS growth for Eurozone is {eps_growth:.1f}%, in line with JPMorgan's thesis.",
                    conviction="HIGH",
                    action="Use dips to build Europe exposure.",
                    target_allocation="Maintain or increase OW in quality EU names.",
                    timeframe="12–24 months",
                    metrics={"eps_growth_2026": eps_growth},
                    timestamp=now,
                )
            )
        elif eps_growth < 8.0:
            signals.append(
                self._build_signal(
                    sid="eps_growth_risk",
                    s_type="WARNING",
                    title="Eurozone EPS Growth Slipping Below Thesis",
                    message=f"2026 EPS growth for Eurozone is {eps_growth:.1f}%, below the 12% thesis anchor.",
                    conviction="MEDIUM",
                    action="Be selective; focus on sectors with visible earnings upgrades.",
                    target_allocation="Reduce beta, tilt towards quality and balance sheet strength.",
                    timeframe="6–12 months",
                    metrics={"eps_growth_2026": eps_growth},
                    timestamp=now,
                )
            )

        # Valuation gap signal
        if pe_gap >= cfg.pe_gap_value_buy:
            signals.append(
                self._build_signal(
                    sid="valuation_gap_value_buy",
                    s_type="BUY",
                    title="EU Valuation Discount vs US at Attractive Levels",
                    message=f"EU trades at a {pe_gap:.1f}x P/E discount vs the US, "
                            f"offering a strong value cushion.",
                    conviction="HIGH",
                    action="Allocate incrementally to EU value and quality cyclicals.",
                    target_allocation="Increase exposure to EU value/financials.",
                    timeframe="6–18 months",
                    metrics={"pe_gap": pe_gap},
                    timestamp=now,
                )
            )
        elif pe_gap <= cfg.pe_gap_warning:
            signals.append(
                self._build_signal(
                    sid="valuation_gap_warning",
                    s_type="INFO",
                    title="Valuation Gap Narrowing",
                    message=f"EU-US P/E gap narrowed to {pe_gap:.1f}x; valuation support is less pronounced.",
                    conviction="LOW",
                    action="Do not rely solely on valuation; focus on earnings revisions.",
                    target_allocation="Keep OW only where earnings momentum is strong.",
                    timeframe="3–9 months",
                    metrics={"pe_gap": pe_gap},
                    timestamp=now,
                )
            )

        # Fragmentation risk
        if fr_de_spread >= cfg.fr_de_high_risk:
            signals.append(
                self._build_signal(
                    sid="fragmentation_high_risk",
                    s_type="RISK_ALERT",
                    title="EU Fragmentation Risk Elevated (FR-DE Spread)",
                    message=f"France–Germany 10Y spread at {fr_de_spread:.0f} bps, above {cfg.fr_de_high_risk:.0f} bps "
                            f"danger zone. Watch EU political/fiscal headlines.",
                    conviction="HIGH",
                    action="Tighten risk limits on France-heavy exposures and high-beta EU periphery.",
                    target_allocation="Shift part of exposure to core Europe and global names.",
                    timeframe="1–6 months",
                    metrics={"fr_de_spread": fr_de_spread},
                    timestamp=now,
                )
            )
        elif fr_de_spread < cfg.fr_de_low_risk:
            signals.append(
                self._build_signal(
                    sid="fragmentation_low_risk",
                    s_type="OPPORTUNITY",
                    title="EU Fragmentation Risk Contained",
                    message=f"France–Germany 10Y spread at {fr_de_spread:.0f} bps, below "
                            f"{cfg.fr_de_low_risk:.0f} bps low-risk threshold.",
                    conviction="MEDIUM",
                    action="More comfortable carrying EU beta and periphery exposure.",
                    target_allocation="Allow higher EU beta within risk budget.",
                    timeframe="6–12 months",
                    metrics={"fr_de_spread": fr_de_spread},
                    timestamp=now,
                )
            )

        # Credit impulse
        if credit_impulse >= cfg.credit_impulse_supportive:
            signals.append(
                self._build_signal(
                    sid="credit_impulse_supportive",
                    s_type="BUY",
                    title="Eurozone Credit Impulse Supports Risk Assets",
                    message=f"Eurozone credit impulse at {credit_impulse:.1f}% of GDP is consistent "
                            f"with improving activity and risk appetite.",
                    conviction="MEDIUM",
                    action="Stay OW EU cyclicals and financials while credit impulse remains positive.",
                    target_allocation="Maintain cyclical tilt in EU book.",
                    timeframe="6–12 months",
                    metrics={"credit_impulse": credit_impulse},
                    timestamp=now,
                )
            )

        return signals

    @staticmethod
    def _build_signal(
        sid: str,
        s_type: str,
        title: str,
        message: str,
        conviction: str,
        action: str,
        target_allocation: str,
        timeframe: str,
        metrics: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Helper to standardize signal dict structure."""
        return {
            "id": sid,
            "type": s_type,  # e.g. BUY / STRONG_BUY / WARNING / RISK_ALERT / INFO / OPPORTUNITY
            "title": title,
            "message": message,
            "conviction": conviction,  # HIGH / MEDIUM / LOW
            "action": action,
            "target_allocation": target_allocation,
            "timeframe": timeframe,
            "metrics": metrics or {},
            "timestamp": timestamp or datetime.utcnow(),
        }

    @staticmethod
    def _is_high_severity_risk(signal: Dict[str, Any]) -> bool:
        """Return True if signal represents a high-severity risk alert."""
        s_type = (signal.get("type") or "").upper()
        return s_type in {"RISK_ALERT"} or "HIGH" in (signal.get("conviction") or "").upper()


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Minimal synthetic data for a quick self-test
    dummy_data = {
        "indices": {
            "relative_performance": -6.5,  # EU underperformed US by 6.5%
        },
        "valuations": {
            "pe_gap": 8.5,  # US - EU
        },
        "macro": {
            "fr_de_spread": 72.0,
        },
        "eps_growth_2026": 13.2,
        "credit_impulse": 3.5,
        "sentiment": {
            "score": 25.0,
        },
    }

    gen = SignalGenerator()
    sigs = gen.generate_signals(dummy_data)
    rec = gen.get_overall_recommendation()

    print("\n=== Signals (head) ===")
    print(pd.DataFrame(sigs).head())

    print("\n=== Overall Recommendation ===")
    print(rec)
