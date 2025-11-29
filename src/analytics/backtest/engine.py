"""
Backtest Engine

JPMorgan-grade backtest wrapper that:
- Fetches market data (free sources only)
- Implements the JPM European thesis timing strategy
- Simulates portfolio equity vs buy-and-hold benchmark
- Computes institutional metrics (return, Sharpe, drawdown, etc.)
- Returns a results dict compatible with Streamlit app + notebooks
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import logging
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Optional imports – engine works even if these fail
# -----------------------------------------------------------------------------
try:
    # Your Yahoo connector (if implemented)
    from src.data.connectors.yahoo import YahooMarketDataConnector  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    YahooMarketDataConnector = None  # type: ignore

try:
    # Transaction cost model (if you created this)
    from src.analytics.backtest.transaction_cost import TransactionCostModel  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TransactionCostModel = None  # type: ignore

try:
    # Central settings
    from config.settings import settings  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    settings = None  # type: ignore

# Fallback: use yfinance directly if your connector is unavailable
try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError("yfinance is required for BacktestEngine when no Yahoo connector is available") from exc


logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """
    Configuration for the backtest engine.

    Attributes:
        start_date: Backtest start date (YYYY-MM-DD).
        end_date: Backtest end date (YYYY-MM-DD) or None = today.
        initial_capital: Starting portfolio value.
        lookback_window: Window (trading days) for relative performance.
        oversold_threshold: Relative underperformance threshold to go 100% long (e.g., -0.10 for -10%).
        take_profit_threshold: Relative outperformance threshold to go 0% long (e.g., +0.05 for +5%).
        rebalance_frequency: 'daily', 'weekly', or 'monthly'.
        transaction_cost_bps: Transaction cost in basis points per unit of turnover.
        slippage_bps: Slippage in basis points per unit of turnover.
        europe_symbol: Yahoo ticker for European equity proxy (default: STOXX Europe 600 ETF EXSA.DE).
        benchmark_symbol: Yahoo ticker for benchmark (default: S&P 500 index).
    """

    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    initial_capital: float = 1_000_000.0

    lookback_window: int = 20
    oversold_threshold: float = -0.10  # -10%
    take_profit_threshold: float = 0.05  # +5%

    rebalance_frequency: str = "daily"  # this implementation is daily; hooks left for future

    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0

    europe_symbol: str = "EXSA.DE"      # STOXX Europe 600 ETF (or your proxy)
    benchmark_symbol: str = "^GSPC"     # S&P 500 index as benchmark


class BacktestEngine:
    """
    High-level backtest engine for the JPM European equity timing strategy.

    Workflow:
        1. Load price data for Europe proxy & benchmark.
        2. Compute daily returns.
        3. Compute 20-day rolling relative performance (Europe - US).
        4. Generate allocation signals (0%, 50%, 100%).
        5. Simulate portfolio equity with transaction costs & slippage.
        6. Compute performance metrics vs benchmark buy-and-hold.

    The engine is intentionally self-contained (no database required) and
    uses free data sources (Yahoo Finance via `yfinance` or your connector).
    """

    def __init__(self, config: Optional[BacktestConfig] = None, start_date: Optional[str] = None) -> None:
        """
        Initialize the backtest engine.

        Args:
            config: Optional BacktestConfig. If not provided, defaults are used.
            start_date: Convenience override for config.start_date.
        """
        self.config: BacktestConfig = config or BacktestConfig()

        if start_date is not None:
            self.config.start_date = start_date

        if self.config.end_date is None:
            self.config.end_date = datetime.today().strftime("%Y-%m-%d")

        self.results: Dict[str, Any] = {}

        # Transaction cost model params (simple embedded logic; optional external model)
        self._tcbps: float = self.config.transaction_cost_bps
        self._slipbps: float = self.config.slippage_bps

        logger.info(
            "BacktestEngine initialized: start=%s end=%s europe=%s bench=%s",
            self.config.start_date,
            self.config.end_date,
            self.config.europe_symbol,
            self.config.benchmark_symbol,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run_backtest(self) -> Dict[str, Any]:
        """
        Run the full backtest and return a results dictionary.

        Returns:
            Dict with keys:
                - 'dates': list[pd.Timestamp]
                - 'strategy_equity': list[float]
                - 'buyhold_equity': list[float]
                - 'strategy_returns': pd.Series
                - 'benchmark_returns': pd.Series
                - 'strategy_metrics': dict
                - 'buyhold_metrics': dict
        """
        logger.info("Starting backtest run...")

        # 1. Load price data
        prices_eu, prices_bench = self._load_price_data(
            self.config.europe_symbol,
            self.config.benchmark_symbol,
            self.config.start_date,
            self.config.end_date,
        )

        # Align & compute returns
        df_prices = pd.concat(
            [prices_eu.rename("europe"), prices_bench.rename("benchmark")],
            axis=1,
        ).dropna()

        if df_prices.empty:
            raise ValueError("No overlapping price data for Europe and benchmark in the given period.")

        returns_eu = df_prices["europe"].pct_change().dropna()
        returns_bench = df_prices["benchmark"].pct_change().dropna()

        # 2. Generate allocation weights based on relative performance
        weights = self._generate_allocations(returns_eu, returns_bench)

        # 3. Simulate strategy equity curve (with basic transaction costs)
        strat_equity, strat_returns = self._simulate_equity(
            returns_eu,
            weights,
            initial_equity=1.0,
        )

        # 4. Benchmark buy-and-hold equity curve
        bench_equity = (1.0 * (1 + returns_eu).cumprod())
        bench_returns = returns_eu.copy()  # for metrics, we treat benchmark as simple B&H on Europe leg

        # 5. Compute performance metrics
        strat_metrics = self._compute_performance_metrics(strat_returns, strat_equity, name="strategy")
        bench_metrics = self._compute_performance_metrics(bench_returns, bench_equity, name="buyhold")

        # 6. Populate results (format expected by Streamlit app + notebooks)
        self.results = {
            "dates": strat_equity.index.to_list(),
            "strategy_equity": strat_equity.values.tolist(),
            "buyhold_equity": bench_equity.values.tolist(),
            "strategy_returns": strat_returns,
            "benchmark_returns": bench_returns,
            "strategy_metrics": strat_metrics,
            "buyhold_metrics": bench_metrics,
        }

        logger.info(
            "Backtest completed: strategy_total_return=%.2f%%, buyhold_total_return=%.2f%%",
            (strat_metrics["total_return"] * 100.0),
            (bench_metrics["total_return"] * 100.0),
        )

        return self.results

    # -------------------------------------------------------------------------
    # Data loading helpers
    # -------------------------------------------------------------------------

    def _load_price_data(
        self,
        europe_symbol: str,
        benchmark_symbol: str,
        start: str,
        end: str,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Load daily close prices for Europe and benchmark.

        Tries:
            1. Your `YahooMarketDataConnector` (if available)
            2. Direct `yfinance` calls

        Returns:
            (europe_prices, benchmark_prices) as pandas Series.
        """
        logger.info(
            "Loading price data from %s to %s for %s (EU) and %s (benchmark)",
            start,
            end,
            europe_symbol,
            benchmark_symbol,
        )

        if YahooMarketDataConnector is not None:
            try:
                connector = YahooMarketDataConnector()
                df_eu = connector.get_history(europe_symbol, start=start, end=end, interval="1d")
                df_bench = connector.get_history(benchmark_symbol, start=start, end=end, interval="1d")

                europe_prices = df_eu["adj_close"] if "adj_close" in df_eu.columns else df_eu["Close"]
                benchmark_prices = df_bench["adj_close"] if "adj_close" in df_bench.columns else df_bench["Close"]

                logger.info("Loaded price data via YahooMarketDataConnector.")
                return europe_prices, benchmark_prices
            except Exception as exc:
                logger.warning("YahooMarketDataConnector failed, falling back to yfinance: %s", exc)

        # Fallback: direct yfinance
        try:
            df = yf.download(
                [europe_symbol, benchmark_symbol],
                start=start,
                end=end,
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                prices_eu = df["Adj Close"][europe_symbol].dropna()
                prices_bench = df["Adj Close"][benchmark_symbol].dropna()
            else:
                # If a single column (rare for multiple symbols), assume 'Adj Close'
                prices_eu = df["Adj Close"].dropna()
                prices_bench = df["Adj Close"].dropna()

            logger.info("Loaded price data via yfinance.")
            return prices_eu, prices_bench
        except Exception as exc:
            logger.error("Failed to load price data from yfinance: %s", exc)
            raise

    # -------------------------------------------------------------------------
    # Strategy logic
    # -------------------------------------------------------------------------

    def _generate_allocations(
        self,
        returns_eu: pd.Series,
        returns_bench: pd.Series,
    ) -> pd.Series:
        """
        Generate allocation weights (0, 0.5, 1.0) based on rolling relative performance.

        Logic (from your thesis/backtest rules):
            - Compute 20-day rolling cumulative returns for Europe & benchmark.
            - Relative performance = Europe_20d - Benchmark_20d
            - If rel_perf < oversold_threshold (e.g., -10%): 100% long Europe
            - If oversold_threshold <= rel_perf <= take_profit_threshold: 50% exposure
            - If rel_perf > take_profit_threshold (e.g., +5%): 0% exposure (take profit)

        Args:
            returns_eu: Daily returns of European index/ETF.
            returns_bench: Daily returns of benchmark (S&P 500 or similar).

        Returns:
            weights: pandas.Series of target allocations in [0.0, 0.5, 1.0].
        """
        df = pd.concat(
            [returns_eu.rename("europe"), returns_bench.rename("benchmark")],
            axis=1,
        ).dropna()

        # Rolling cumulative 20-day returns
        window = self.config.lookback_window

        def _cum_ret(x: np.ndarray) -> float:
            return float((1.0 + x).prod() - 1.0)

        roll_eu = df["europe"].rolling(window).apply(_cum_ret, raw=True)
        roll_bench = df["benchmark"].rolling(window).apply(_cum_ret, raw=True)

        rel_perf = (roll_eu - roll_bench).rename("rel_perf")

        weights = pd.Series(index=df.index, dtype=float)

        prev_weight = 0.0
        for ts, rel in rel_perf.items():
            if np.isnan(rel):
                weights.at[ts] = prev_weight
                continue

            if rel < self.config.oversold_threshold:
                w = 1.0
            elif rel <= self.config.take_profit_threshold:
                w = 0.5
            else:
                w = 0.0

            weights.at[ts] = w
            prev_weight = w

        weights.name = "weight"
        logger.info(
            "Generated allocation weights (min=%.2f, max=%.2f)",
            float(weights.min()),
            float(weights.max()),
        )
        return weights

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def _simulate_equity(
        self,
        returns: pd.Series,
        weights: pd.Series,
        initial_equity: float = 1.0,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Simulate strategy equity using time-varying weights and simple transaction costs.

        Transaction cost model:
            - Turnover_t = |w_t - w_{t-1}|
            - Cost_rate = (tc_bps + slip_bps) / 10_000
            - Daily strategy return = w_t * asset_ret_t - Turnover_t * Cost_rate

        Args:
            returns: Daily asset returns (Europe).
            weights: Daily weights in asset.
            initial_equity: Starting normalized equity (default 1.0).

        Returns:
            (equity_curve, strategy_returns) as pandas.Series.
        """
        df = pd.concat(
            [returns.rename("ret"), weights.rename("weight")],
            axis=1,
        ).dropna()

        equity = pd.Series(index=df.index, dtype=float)
        strat_ret = pd.Series(index=df.index, dtype=float)

        equity.iloc[0] = initial_equity
        prev_weight = df["weight"].iloc[0]

        tc_rate = (self._tcbps + self._slipbps) / 10_000.0

        for i, (ts, row) in enumerate(df.iterrows()):
            w = float(row["weight"])
            r = float(row["ret"])

            if i == 0:
                # First step – assume no immediate transaction costs (already positioned)
                turnover = 0.0
            else:
                turnover = abs(w - prev_weight)

            daily_cost = turnover * tc_rate
            daily_return = (w * r) - daily_cost

            if i == 0:
                equity.iloc[i] = initial_equity * (1.0 + daily_return)
            else:
                equity.iloc[i] = equity.iloc[i - 1] * (1.0 + daily_return)

            strat_ret.iloc[i] = daily_return
            prev_weight = w

        equity.name = "strategy_equity"
        strat_ret.name = "strategy_return"

        return equity, strat_ret

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_performance_metrics(
        returns: pd.Series,
        equity: pd.Series,
        name: str,
        ann_factor: int = 252,
    ) -> Dict[str, float]:
        """
        Compute core performance & risk statistics.

        Metrics:
            - total_return
            - annualized_return
            - volatility
            - sharpe_ratio (rf ~ 0)
            - max_drawdown
            - win_rate

        Args:
            returns: Daily return series.
            equity: Equity curve corresponding to returns.
            name: Label for logging.
            ann_factor: Annualization factor (default 252 trading days).

        Returns:
            Dict of metrics.
        """
        r = returns.dropna()
        eq = equity.dropna()

        if r.empty or eq.empty:
            logger.warning("No data for performance metrics (%s).", name)
            return {
                "total_return": float("nan"),
                "annualized_return": float("nan"),
                "volatility": float("nan"),
                "sharpe_ratio": float("nan"),
                "max_drawdown": float("nan"),
                "win_rate": float("nan"),
            }

        mean_daily = r.mean()
        vol_daily = r.std()

        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        ann_return = float((1.0 + mean_daily) ** ann_factor - 1.0)
        ann_vol = float(vol_daily * np.sqrt(ann_factor))
        sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")

        running_max = eq.cummax()
        drawdown = eq / running_max - 1.0
        max_dd = float(drawdown.min())

        win_rate = float((r > 0).mean())

        metrics = {
            "total_return": total_return,
            "annualized_return": ann_return,
            "volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
        }

        logger.info(
            "Metrics (%s): total=%.2f%% ann=%.2f%% vol=%.2f%% sharpe=%.2f maxDD=%.2f%% win_rate=%.1f%%",
            name,
            total_return * 100.0,
            ann_return * 100.0,
            ann_vol * 100.0,
            sharpe if not np.isnan(sharpe) else float("nan"),
            max_dd * 100.0,
            win_rate * 100.0,
        )

        return metrics


if __name__ == "__main__":
    # Simple smoke test (runs with free Yahoo data)
    logging.basicConfig(level=logging.INFO)
    engine = BacktestEngine(start_date="2020-01-01")
    res = engine.run_backtest()

    print("\n=== BACKTEST SUMMARY ===")
    print("Strategy total return:  {:.2%}".format(res["strategy_metrics"]["total_return"]))
    print("Buy & hold total return:{:.2%}".format(res["buyhold_metrics"]["total_return"]))
    print("Strategy Sharpe:        {:.2f}".format(res["strategy_metrics"]["sharpe_ratio"]))
    print("Buy & hold Sharpe:      {:.2f}".format(res["buyhold_metrics"]["sharpe_ratio"]))
