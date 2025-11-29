"""
Excel Exporter

Generates institutional-style Excel reports for the JPMorgan
European Equity Dashboard, including:

- Portfolio snapshot (positions, weights, prices, risk metrics)
- Backtest results (equity curve, trades, performance metrics)
- Factor exposures and factor returns

Uses openpyxl as the engine via pandas ExcelWriter.
All exports are designed to be PowerBI / CIO-ready.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExcelExporterConfig:
    """
    Configuration for Excel exports.

    Attributes:
        date_format: Excel date format string.
        float_format: Default numeric format.
        percent_format: Percentage format.
        currency_format: Currency format (EUR by default).
    """

    date_format: str = "yyyy-mm-dd"
    float_format: str = "#,##0.0000"
    percent_format: str = "0.00%"
    currency_format: str = "€#,##0.00"


class ExcelExporter:
    """
    Excel Exporter for the JPMorgan European Equity Dashboard.

    Typical usage:

        exporter = ExcelExporter()

        exporter.export_portfolio_snapshot(
            weights=weights_dict,
            prices=prices_series,
            risk_metrics=risk_dict,
            file_path="exports/portfolio_snapshot.xlsx",
        )

        exporter.export_backtest_results(
            equity_curve=equity_df,
            trades=trades_df,
            performance_metrics=metrics_dict,
            file_path="exports/backtest_report.xlsx",
        )

        exporter.export_factor_exposures(
            factor_exposures=exposures_df,
            factor_returns=factor_ret_df,
            file_path="exports/factors.xlsx",
        )
    """

    def __init__(self, config: Optional[ExcelExporterConfig] = None) -> None:
        self.config = config or ExcelExporterConfig()
        logger.info("ExcelExporter initialized with config: %s", self.config)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def export_portfolio_snapshot(
        self,
        weights: Mapping[str, float],
        prices: Optional[Mapping[str, float]] = None,
        risk_metrics: Optional[Mapping[str, float]] = None,
        meta: Optional[Mapping[str, Any]] = None,
        file_path: str | Path = "portfolio_snapshot.xlsx",
    ) -> Path:
        """
        Export a portfolio snapshot to Excel.

        Sheets created:
            - Summary: high-level info, risk metrics
            - Positions: symbol, weight, price, notional (if price & total value given)
            - Meta: additional context (optional)

        Args:
            weights: Dict symbol -> portfolio weight.
            prices: Optional dict symbol -> price.
            risk_metrics: Optional dict of risk metrics (e.g. var_95, sharpe, etc.).
            meta: Optional dict of metadata (strategy name, as-of date, etc.).
            file_path: Output Excel path.

        Returns:
            Path to the created Excel file.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting portfolio snapshot to %s", file_path)

        weights_series = pd.Series(weights, name="weight").sort_index()
        prices_series = pd.Series(prices or {}, name="price").reindex(weights_series.index)

        # Build positions DataFrame
        positions_df = pd.concat([weights_series, prices_series], axis=1)
        # Notional is left to user to compute externally if needed, but we can approximate
        # using total portfolio value in meta if provided.
        portfolio_value = float(meta.get("portfolio_value", 1.0)) if meta else 1.0
        positions_df["notional"] = positions_df["weight"].fillna(0.0) * portfolio_value

        # Risk metrics table
        risk_df = pd.DataFrame(
            [
                {"Metric": k, "Value": float(v)}
                for k, v in (risk_metrics or {}).items()
            ]
        ).sort_values("Metric") if risk_metrics else pd.DataFrame(
            columns=["Metric", "Value"]
        )

        # Meta table
        meta_df = pd.DataFrame(
            [
                {"Key": k, "Value": v}
                for k, v in (meta or {}).items()
            ]
        ) if meta else pd.DataFrame(columns=["Key", "Value"])

        # Write Excel
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Summary sheet
            summary_df = pd.DataFrame(
                {
                    "Item": ["As Of", "Total Positions", "Portfolio Value"],
                    "Value": [
                        meta.get("as_of", datetime.utcnow().isoformat()) if meta else datetime.utcnow().isoformat(),
                        len(weights_series),
                        portfolio_value,
                    ],
                }
            )
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            if not risk_df.empty:
                # Append risk metrics below summary in same sheet
                start_row = len(summary_df) + 3
                risk_df.to_excel(writer, sheet_name="Summary", index=False, startrow=start_row)

            # Positions sheet
            positions_df.to_excel(writer, sheet_name="Positions", index_label="symbol")

            # Meta sheet
            if not meta_df.empty:
                meta_df.to_excel(writer, sheet_name="Meta", index=False)

            workbook = writer.book
            # Format summary & positions sheets
            self._auto_format_sheet(workbook["Summary"])
            self._auto_format_sheet(workbook["Positions"])
            if "Meta" in workbook.sheetnames:
                self._auto_format_sheet(workbook["Meta"])

        logger.info("Portfolio snapshot export completed: %s", file_path)
        return file_path

    def export_backtest_results(
        self,
        equity_curve: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None,
        performance_metrics: Optional[Mapping[str, float]] = None,
        file_path: str | Path = "backtest_report.xlsx",
    ) -> Path:
        """
        Export backtest results to Excel.

        Sheets created:
            - Performance: metrics and summary
            - EquityCurve: time series of equity curves
            - Trades: executed trades (optional)

        Args:
            equity_curve: DataFrame with Date index or 'date' column, and columns like
                          ['strategy_equity', 'benchmark_equity'].
            trades: Optional DataFrame of trades with columns like
                    ['date', 'symbol', 'action', 'quantity', 'price', 'notional'].
            performance_metrics: Dict of performance metrics.
            file_path: Output Excel path.

        Returns:
            Path to the created Excel file.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting backtest results to %s", file_path)

        # Normalize equity_curve index
        eq_df = equity_curve.copy()
        if "date" in eq_df.columns and not isinstance(eq_df.index, pd.DatetimeIndex):
            eq_df["date"] = pd.to_datetime(eq_df["date"])
            eq_df = eq_df.set_index("date")
        elif isinstance(eq_df.index, pd.DatetimeIndex):
            eq_df = eq_df.copy()
        else:
            # no date info – just keep as is
            pass

        perf_df = pd.DataFrame(
            [
                {"Metric": k, "Value": float(v)}
                for k, v in (performance_metrics or {}).items()
            ]
        ).sort_values("Metric") if performance_metrics else pd.DataFrame(
            columns=["Metric", "Value"]
        )

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Performance sheet
            perf_df.to_excel(writer, sheet_name="Performance", index=False)

            # Equity curve sheet
            eq_df.to_excel(writer, sheet_name="EquityCurve")

            # Trades sheet
            if trades is not None and not trades.empty:
                trades.to_excel(writer, sheet_name="Trades", index=False)

            workbook = writer.book
            if "Performance" in workbook.sheetnames:
                self._auto_format_sheet(workbook["Performance"])
            if "EquityCurve" in workbook.sheetnames:
                self._auto_format_sheet(workbook["EquityCurve"])
            if "Trades" in workbook.sheetnames:
                self._auto_format_sheet(workbook["Trades"])

        logger.info("Backtest results export completed: %s", file_path)
        return file_path

    def export_factor_exposures(
        self,
        factor_exposures: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None,
        file_path: str | Path = "factor_report.xlsx",
    ) -> Path:
        """
        Export factor exposures and factor returns to Excel.

        Sheets created:
            - Exposures: cross-section of exposures by asset & factor
            - FactorReturns: time-series factor returns (optional)

        Args:
            factor_exposures: DataFrame with index as assets and columns as factors,
                              or MultiIndex (asset, date) vs factors.
            factor_returns: Optional DataFrame with Date index and factors as columns.
            file_path: Output Excel path.

        Returns:
            Path to the created Excel file.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting factor exposures to %s", file_path)

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            factor_exposures.to_excel(writer, sheet_name="Exposures")

            if factor_returns is not None and not factor_returns.empty:
                factor_returns.to_excel(writer, sheet_name="FactorReturns")

            workbook = writer.book
            if "Exposures" in workbook.sheetnames:
                self._auto_format_sheet(workbook["Exposures"])
            if "FactorReturns" in workbook.sheetnames:
                self._auto_format_sheet(workbook["FactorReturns"])

        logger.info("Factor exposures export completed: %s", file_path)
        return file_path

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _auto_format_sheet(self, ws) -> None:
        """
        Apply basic formatting to a worksheet:
        - Auto-fit column widths
        - Apply numeric formats heuristically
        """
        # Auto column width based on max length in each column
        for column_cells in ws.columns:
            max_length = 0
            col = column_cells[0].column_letter
            for cell in column_cells:
                try:
                    value = cell.value
                    if value is None:
                        cell_length = 0
                    else:
                        cell_length = len(str(value))
                    if cell_length > max_length:
                        max_length = cell_length
                except Exception:
                    # Fallback if any weird type
                    continue
            adjusted_width = min(max_length + 2, 40)
            ws.column_dimensions[col].width = adjusted_width

        # Apply formats heuristically to numeric columns
        for row in ws.iter_rows(min_row=2):  # skip header
            for cell in row:
                if isinstance(cell.value, (int, float, np.number)):
                    # Heuristic: treat values between -1 and 1 as percent
                    if -1.0 <= float(cell.value) <= 1.0 and abs(float(cell.value)) < 0.5:
                        cell.number_format = self.config.float_format
                    else:
                        cell.number_format = self.config.float_format

    # -------------------------------------------------------------------------
    # Convenience static methods
    # -------------------------------------------------------------------------

    @staticmethod
    def build_simple_equity_curve(
        dates: pd.DatetimeIndex,
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Utility to build an equity curve DataFrame from daily returns.

        Args:
            dates: DatetimeIndex for the period.
            strategy_returns: Daily strategy returns (Series aligned with dates).
            benchmark_returns: Daily benchmark returns (optional).

        Returns:
            DataFrame with columns: ['strategy_equity', 'benchmark_equity' (optional)].
        """
        df = pd.DataFrame(index=dates)
        df["strategy_return"] = strategy_returns.reindex(dates).fillna(0.0)
        df["strategy_equity"] = (1.0 + df["strategy_return"]).cumprod()

        if benchmark_returns is not None:
            df["benchmark_return"] = benchmark_returns.reindex(dates).fillna(0.0)
            df["benchmark_equity"] = (1.0 + df["benchmark_return"]).cumprod()

        return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    exporter = ExcelExporter()

    # Fake data for quick smoke test
    weights = {"SXXP": 0.4, "SX7E": 0.3, "SX8E": 0.3}
    prices = {"SXXP": 450.0, "SX7E": 90.0, "SX8E": 110.0}
    risk_metrics = {
        "total_return": 0.25,
        "annualized_vol": 0.18,
        "sharpe_ratio": 1.3,
        "var_95": 0.03,
    }
    meta = {
        "as_of": datetime.utcnow().strftime("%Y-%m-%d"),
        "strategy_name": "EU Equity Overweight",
        "portfolio_value": 1_000_000,
    }

    exporter.export_portfolio_snapshot(
        weights=weights,
        prices=prices,
        risk_metrics=risk_metrics,
        meta=meta,
        file_path="exports/example_portfolio_snapshot.xlsx",
    )

    # Backtest smoke
    idx = pd.date_range("2020-01-01", periods=250, freq="B")
    strat_ret = pd.Series(np.random.normal(0.0005, 0.01, len(idx)), index=idx)
    bench_ret = pd.Series(np.random.normal(0.0004, 0.009, len(idx)), index=idx)
    eq_df = ExcelExporter.build_simple_equity_curve(idx, strat_ret, bench_ret)
    perf = {
        "total_return": eq_df["strategy_equity"].iloc[-1] - 1,
        "annualized_return": (1 + eq_df["strategy_equity"].iloc[-1] - 1) ** (252 / len(eq_df)) - 1,
        "sharpe_ratio": 1.5,
    }

    exporter.export_backtest_results(
        equity_curve=eq_df,
        trades=pd.DataFrame(
            [
                {
                    "date": idx[10],
                    "symbol": "SXXP",
                    "action": "BUY",
                    "quantity": 100,
                    "price": 430.0,
                    "notional": 43_000.0,
                }
            ]
        ),
        performance_metrics=perf,
        file_path="exports/example_backtest_report.xlsx",
    )

    # Factor exposures smoke
    exposures = pd.DataFrame(
        {
            "Value": [0.3, -0.1, 0.2],
            "Momentum": [0.1, 0.5, -0.2],
            "Quality": [0.2, 0.1, 0.0],
        },
        index=["SXXP", "SX7E", "SX8E"],
    )
    factor_ret = pd.DataFrame(
        {
            "Value": np.random.normal(0.0001, 0.001, len(idx)),
            "Momentum": np.random.normal(0.0001, 0.001, len(idx)),
            "Quality": np.random.normal(0.0001, 0.001, len(idx)),
        },
        index=idx,
    )

    exporter.export_factor_exposures(
        factor_exposures=exposures,
        factor_returns=factor_ret,
        file_path="exports/example_factor_report.xlsx",
    )

    logger.info("Smoke tests completed.")
