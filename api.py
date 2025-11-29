"""
API Layer for JPMorgan European Equity Dashboard
-------------------------------------------------

Provides:
- /health        : System health
- /data/*        : Market, macro, news, signals
- /signals/*     : Signal generation
- /sentiment     : FinBERT sentiment scoring
- /backtest      : Run backtest
- /factors       : Factor analysis
- /portfolio     : Optimization, risk budgeting, rebalance
- /reports       : PDF & Excel report generation
- /ws/stream     : Real-time data websocket

Architecture:
- FastAPI for REST
- WebSocket for real-time updates
- Integrates src/data, src/analytics, src/portfolio, src/reporting
"""

from __future__ import annotations

import logging
import traceback
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# -------------------------
# Internal modules
# -------------------------
from src.data.connectors.yahoo import YahooMarketData
from src.data.connectors.fred import FREDClient
from src.data.connectors.newsapi import NewsAPIClient

from src.data.processors.validator import DataValidator
from src.data.processors.cleaner import DataCleaner
from src.data.processors.normalizer import DataNormalizer
from src.data.processors.aggregator import DataAggregator

from src.analytics.signals.generator import SignalGenerator
from src.analytics.sentiment.finbert import FinBERTSentiment
from src.analytics.backtest.engine import BacktestEngine
from src.analytics.factors.analyzer import FactorAnalyzer

from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.rebalancer import PortfolioRebalancer
from src.portfolio.risk_budget import RiskBudgetAllocator

from src.monitoring.health_check import SystemHealthChecker
from src.monitoring.data_quality import DataQualityChecker

from src.reporting.pdf_generator import PDFReportGenerator
from src.reporting.excel_exporter import ExcelExporter

from src.utils.logger import get_logger


# ============================================================================
# FastAPI Initialization
# ============================================================================

logger = get_logger("API")

app = FastAPI(
    title="JPMorgan European Equity Dashboard API",
    version="1.0.0",
    description="REST & WebSocket API powering the JPMorgan European Equity Thesis Monitor",
)


# ============================================================================
# CORS (Allow your Streamlit frontend)
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Dependency Injection for modules
# ============================================================================

def get_modules():
    """Initialize heavy modules once."""
    return {
        "signals": SignalGenerator(),
        "sentiment": FinBERTSentiment(),
        "backtest": BacktestEngine(start_date="2020-01-01"),
        "factors": FactorAnalyzer(),
        "optimizer": PortfolioOptimizer(),
        "rebalancer": PortfolioRebalancer(),
        "risk_budget": RiskBudgetAllocator(),
        "pdf": PDFReportGenerator(),
        "excel": ExcelExporter(),
        "health": SystemHealthChecker(),
        "dq": DataQualityChecker(),
    }


def get_connectors():
    return {
        "yahoo": YahooMarketData(),
        "fred": FREDClient(),
        "news": NewsAPIClient(),
    }


def get_processors():
    return {
        "validator": DataValidator(),
        "cleaner": DataCleaner(),
        "normalizer": DataNormalizer(),
        "aggregator": DataAggregator(),
    }


# ============================================================================
# Utility
# ============================================================================

def safe_execute(fn, **kwargs):
    """Unified error wrapper."""
    try:
        result = fn(**kwargs) if kwargs else fn()
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error("API error: %s\n%s", str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH ENDPOINTS
# ============================================================================

@app.get("/health", tags=["System"])
def system_health(modules=Depends(get_modules)):
    return safe_execute(modules["health"].check_system)


@app.get("/health/data", tags=["System"])
def data_quality(modules=Depends(get_modules), connectors=Depends(get_connectors)):
    return safe_execute(modules["dq"].check_data_quality, connectors=connectors)


# ============================================================================
# DATA ENDPOINTS
# ============================================================================

@app.get("/data/prices", tags=["Data"])
def get_prices(ticker: str, connectors=Depends(get_connectors)):
    return safe_execute(connectors["yahoo"].get_price_history, ticker=ticker)


@app.get("/data/macro", tags=["Data"])
def get_macro(series_id: str, connectors=Depends(get_connectors)):
    return safe_execute(connectors["fred"].get_series, series_id=series_id)


@app.get("/data/news", tags=["Data"])
def get_news(keyword: str, connectors=Depends(get_connectors)):
    return safe_execute(connectors["news"].search_news, query=keyword)


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

@app.post("/signals/generate", tags=["Signals"])
def generate_signals(payload: Dict[str, Any], modules=Depends(get_modules)):
    return safe_execute(modules["signals"].generate_signals, data=payload)


@app.post("/signals/overall", tags=["Signals"])
def overall_recommendation(payload: Dict[str, Any], modules=Depends(get_modules)):
    return safe_execute(modules["signals"].get_overall_recommendation, data=payload)


# ============================================================================
# SENTIMENT ANALYSIS
# ============================================================================

@app.post("/sentiment", tags=["Sentiment"])
def sentiment_analysis(payload: Dict[str, Any], modules=Depends(get_modules)):
    articles: List[str] = payload.get("articles", [])
    return safe_execute(modules["sentiment"].score_articles, articles=articles)


# ============================================================================
# FACTOR ANALYSIS
# ============================================================================

@app.post("/factors/analyze", tags=["Factors"])
def analyze_factors(payload: Dict[str, Any], modules=Depends(get_modules)):
    prices = payload.get("prices", {})
    return safe_execute(modules["factors"].compute_factor_exposures, price_data=prices)


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

@app.post("/backtest/run", tags=["Backtest"])
def run_backtest(payload: Dict[str, Any], modules=Depends(get_modules)):
    return safe_execute(modules["backtest"].run_backtest, **payload)


# ============================================================================
# PORTFOLIO MANAGEMENT
# ============================================================================

@app.post("/portfolio/optimize", tags=["Portfolio"])
def optimize_portfolio(payload: Dict[str, Any], modules=Depends(get_modules)):
    return safe_execute(modules["optimizer"].optimize, **payload)


@app.post("/portfolio/rebalance", tags=["Portfolio"])
def rebalance(payload: Dict[str, Any], modules=Depends(get_modules)):
    return safe_execute(modules["rebalancer"].rebalance, **payload)


@app.post("/portfolio/risk-budget", tags=["Portfolio"])
def risk_budget(payload: Dict[str, Any], modules=Depends(get_modules)):
    return safe_execute(modules["risk_budget"].allocate, **payload)


# ============================================================================
# REPORTS
# ============================================================================

@app.post("/reports/pdf", tags=["Reports"])
def generate_pdf(payload: Dict[str, Any], modules=Depends(get_modules)):
    path = modules["pdf"].create_portfolio_report(payload)
    return FileResponse(path, filename="jpm_report.pdf")


@app.post("/reports/excel", tags=["Reports"])
def generate_excel(payload: Dict[str, Any], modules=Depends(get_modules)):
    path = modules["excel"].export_to_excel(payload)
    return FileResponse(path, filename="jpm_data.xlsx")


# ============================================================================
# REAL-TIME WEBSOCKET STREAM
# ============================================================================

@app.websocket("/ws/stream")
async def market_stream(ws: WebSocket):
    await ws.accept()

    yahoo = YahooMarketData()

    try:
        while True:
            prices = yahoo.get_realtime_prices(["^STOXX50E", "^GSPC"])
            await ws.send_json(
                {
                    "status": "ok",
                    "data": prices,
                }
            )
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

    except Exception as e:
        logger.error("WebSocket error: %s", e)
        await ws.close()


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", tags=["Root"])
def root():
    return {"message": "JPMorgan European Equity API is running."}
