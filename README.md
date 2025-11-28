# 📊 JPMorgan European Equity Thesis Monitor

> A real-time, institutional-grade dashboard tracking JPMorgan’s European equity **overweight** thesis

[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

This project is a **JPMorgan-grade European equity monitoring system**, designed to validate and continuously track the 2024+ **European equity overweight** thesis.

It combines:

- Multi-source **market & macro data**
- **Quant signals** (momentum, value, macro regimes)
- **Risk & attribution analytics**
- **Backtesting & portfolio simulation**
- Production patterns (MongoDB, TimescaleDB, Redis, REST API, WebSockets)

All using **open-source / free data sources only** (yfinance, FRED, ECB SDW, NewsAPI, etc.), so it can be run by **students, researchers, and quants** without paid terminals.

We answer three core questions:

1. **Is the thesis working?** – CIO View  
2. **Where is it working?** – PM View  
3. **Why is it working (or not)?** – Strategist / Macro View  

---

## ✨ Key Features

### 📈 Real-Time European vs US Monitoring

- STOXX Europe 600 vs S&P 500 performance (absolute & relative)
- Sector and thematic basket performance:
  - GRANOLAS (European mega-caps)
  - EU Banks
  - EU Defense
  - Fiscal beneficiaries
- Rolling KPIs:
  - Relative performance (3M, 6M, 12M)
  - Drawdown, volatility, Sharpe

### 🧠 Quant & Factor Analytics

- **Signals**
  - Momentum: 1M / 3M / 6M / 12M, risk-adjusted
  - Value: PE, PB, PS, Dividend Yield, Composite
  - Macro regime scores (growth, inflation, policy, curve)
- **Attribution**
  - Brinson sector / asset attribution
  - Factor attribution (e.g. Fama-French style)
  - Sector & factor tilts vs benchmark

### 🛡️ Risk Analytics (Institutional-Grade)

- Value at Risk (VaR) – historical, parametric
- Conditional VaR (Expected Shortfall)
- Volatility & max drawdown
- Beta, tracking error, information ratio
- Scenario & stress testing (e.g. 2008, COVID, EU fragmentation shocks)

### 🧪 Backtesting & Portfolio Simulation

- Daily backtest engine (2020–present)
- Configurable:
  - Rebalancing frequency
  - Transaction costs & slippage
  - Leverage & shorting constraints
- Outputs:
  - Equity curve
  - Risk-adjusted metrics (Sharpe, Sortino)
  - Exposure and turnover statistics

### 🔔 Monitoring & Alerts (Planned / Extensible)

- Threshold-based alerts on:
  - EU vs US underperformance
  - Spread levels (e.g. FR–DE 10Y)
  - Macro regime switches
- Email alert integration via SMTP (e.g. Gmail)

---

## 🧱 High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                      │
│  - Streamlit Dashboard (app.py)                            │
│  - FastAPI REST API (api.py)                               │
│  - WebSocket Server for live updates (websocket_server.py) │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                     │
│  - Signals: momentum, value, macro                         │
│  - Risk: VaR/CVaR, stress tests                            │
│  - Attribution: sector, factor, returns                    │
│  - Backtest: portfolio simulation, transaction costs       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                        DATA LAYER                           │
│  - MongoDB       → Documents (snapshots, signals)          │
│  - TimescaleDB   → Time series (macro, factors)            │
│  - Redis         → Cache (latest quotes, signals)          │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│                    DATA INGESTION LAYER                     │
│  - yfinance       → Indices & equities                     │
│  - FRED           → Macro series                           │
│  - ECB SDW        → Euro-area statistics                   │
│  - NewsAPI        → Headlines for sentiment                │
│  - Validation / Cleaning / Aggregation                     │
└─────────────────────────────────────────────────────────────┘

🗂 Project Structure (Current Target)
JPMorganChase/
│
├── app.py                     # Streamlit dashboard (CIO/PM/Strategist views)
├── api.py                     # FastAPI REST API
├── websocket_server.py        # WebSocket real-time server
│
├── README.md                  # Project documentation (this file)
├── ARCHITECTURE.md            # Detailed system architecture
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                    # MIT License
├── .env.example               # Environment variables template
├── .gitignore
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # MongoDB + TimescaleDB + Redis + API
├── Dockerfile                 # App container
│
├── config/
│   ├── __init__.py
│   ├── settings.py            # Centralized settings (Pydantic)
│   ├── data_sources.yml       # Ticker mappings & sources
│   └── thresholds.yml         # Alert thresholds
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── connectors/
│   │   │   ├── __init__.py
│   │   │   ├── yahoo.py       # yfinance wrapper (done)
│   │   │   ├── fred.py        # FRED client (planned)
│   │   │   ├── ecb_sdw.py     # ECB SDW client (planned)
│   │   │   └── newsapi.py     # News connector (planned)
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── market_data.py # Market data schemas (done)
│   │   │   ├── portfolio.py   # Portfolio schemas (done)
│   │   │   └── signals.py     # Signal schemas (done)
│   │   └── repository/
│   │       ├── __init__.py
│   │       ├── mongodb_repo.py    # MongoDB ops (done)
│   │       ├── timescale_repo.py  # TimescaleDB ops (done)
│   │       └── redis_cache.py     # Redis caching (done)
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── signals/
│   │   │   ├── __init__.py
│   │   │   ├── momentum.py        # Momentum signals (done)
│   │   │   ├── value.py           # Value signals (done)
│   │   │   └── macro.py           # Macro signals (done)
│   │   ├── risk/
│   │   │   ├── __init__.py
│   │   │   └── risk_analytics.py  # VaR/CVaR, Sharpe, stress (done)
│   │   ├── attribution/
│   │   │   ├── __init__.py
│   │   │   ├── returns.py         # Brinson attribution (done)
│   │   │   ├── factors.py         # Factor attribution (done)
│   │   │   └── sector.py          # Sector attribution (done)
│   │   ├── backtest/
│   │   │   ├── __init__.py
│   │   │   ├── portfolio_sim.py   # Portfolio simulator (done)
│   │   │   └── transaction_cost.py# Transaction cost model (done)
│   │   ├── sentiment/             # (planned)
│   │   └── factors/               # (planned)
│   │
│   └── utils/                     # Logging, dates, math (planned)
│
├── tests/                         # Unit & integration tests (planned)
├── notebooks/                     # Research & exploration (planned)
├── scripts/                       # Setup & backfill scripts (planned)
├── data/                          # Raw/processed/cache/exports
└── monitoring/                    # Prometheus/Grafana configs (planned)

🚀 Quick Start
1️⃣ Prerequisites

Python 3.11+

pip (Python package manager)

Optional but recommended:

Docker & docker-compose

MongoDB, TimescaleDB, Redis (if not using Docker)

2️⃣ Install Dependencies
git clone https://github.com/yourusername/JPMorganChase.git
cd JPMorganChase

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

3️⃣ Configure Environment
cp .env.example .env
# Edit .env and add your keys


Minimal .env:

ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

FRED_API_KEY=your_fred_key
NEWSAPI_KEY=your_newsapi_key
ALPHA_VANTAGE_KEY=demo   # or your key

MONGODB_URI=mongodb://localhost:27017/jpm_dashboard
POSTGRES_URI=postgresql://jpm_user:password@localhost:5432/jpm_timeseries
REDIS_URI=redis://localhost:6379/0

4️⃣ Run via Streamlit
streamlit run app.py


Then open: http://localhost:8501

5️⃣ (Optional) Run Full Stack via Docker
docker-compose up --build


This will start:

MongoDB

TimescaleDB

Redis

FastAPI API

Streamlit dashboard

🔑 Free Data Sources & API Keys
Required

FRED – macro & rates

Get key: https://fred.stlouisfed.org/docs/api/api_key.html

.env → FRED_API_KEY=your_key_here

NewsAPI – headlines for sentiment

Get key: https://newsapi.org/register

Free: 100 requests/day

.env → NEWSAPI_KEY=your_key_here

Optional (Email Alerts)

Gmail SMTP

SMTP_EMAIL=your@gmail.com
SMTP_PASSWORD=your_app_password  # Gmail App Password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

📊 Main Dashboard Views (Planned UX)
1. 📈 CIO View – “Is the Thesis Working?”

STOXX 600 vs S&P 500 relative performance (3M, 6M, 12M)

EU vs US forward P/E spread

FR–DE 10Y yield spread (fragmentation risk proxy)

Macro regime score for Europe (growth/inflation/policy/curve)

High-level risk metrics (vol, drawdown, VaR)

2. 💼 PM View – “Where Is It Working?”

Sector performance heatmap / treemap (STOXX 600 sectors)

Thematic baskets:

GRANOLAS

EU defense

EU banks

Fiscal beneficiaries

Contribution to relative performance by sector & theme

3. 🌍 Strategist View – “Why Is It Working?”

Macro indicators:

Eurozone GDP / PMI

Inflation & core inflation

Yield curve slope

Credit spreads (proxy via indices)

MacroSignal (risk-on vs risk-off) summary

Scenario analysis (e.g., tariff shock, growth slowdown)

4. 🧪 Backtest Performance

Backtest vs benchmark since 2020

Equity curve, drawdowns

Sharpe, Sortino, VaR

Trade & turnover statistics

🧪 Usage Examples (Code)

These examples assume you run them from the repo root with a configured environment.

1️⃣ Load Market Data via Repository + Connector
from datetime import date
import pandas as pd

from src.data.connectors.yahoo import YahooMarketDataConnector
from src.data.models.market_data import MarketDataRequest
from src.data.repository.mongodb_repo import MongoDBRepository
from config.settings import settings

# Connector (yfinance)
connector = YahooMarketDataConnector(delay_seconds=settings.YAHOO_FINANCE_DELAY)

# Request STOXX 600 & S&P 500 (proxy tickers)
req = MarketDataRequest(
    symbols=["^STOXX50E", "^GSPC"],
    start_date=date(2020, 1, 1),
    end_date=date.today(),
)

response = connector.fetch_market_data(req)

# Persist to MongoDB
mongo = MongoDBRepository(uri=settings.MONGODB_URI)
for symbol, series in response.series.items():
    mongo.save_price_series(series)

2️⃣ Generate Momentum Signals
import pandas as pd
from src.analytics.signals.momentum import MomentumSignalEngine
from src.data.models.market_data import HistoricalPriceSeries

engine = MomentumSignalEngine()

# Suppose 'series' is HistoricalPriceSeries from MongoDB or connector
df = series.to_dataframe()
signals = engine.compute_momentum_signals(df=df, as_of=df.index[-1], symbol=series.symbol)

for name, sp in signals.items():
    print(name, sp.value, sp.direction, sp.strength)

3️⃣ Run a Simple Portfolio Backtest
import pandas as pd
from src.analytics.backtest.portfolio_sim import PortfolioSimulator, BacktestConfig

prices = ...  # DataFrame [dates x symbols]

# Simple equal-weight rebalance monthly
rebalance_dates = prices.resample("M").last().index
target_weights = {
    dt: pd.Series(1.0 / len(prices.columns), index=prices.columns)
    for dt in rebalance_dates
}

config = BacktestConfig(initial_nav=100.0, rebalance_frequency="M")
sim = PortfolioSimulator(config=config)
series = sim.run_backtest(prices=prices, target_weights=target_weights)

df_perf = series.to_dataframe()
print(df_perf.tail())

4️⃣ Risk Analytics on a Strategy
import numpy as np
from src.analytics.risk.risk_analytics import RiskAnalytics

returns = df_perf["return_daily"]  # from PortfolioPerformanceSeries
benchmark_returns = ...            # Series of benchmark daily returns

risk = RiskAnalytics(confidence_level=0.95)
report = risk.generate_risk_report(returns, benchmark_returns)
print(report)

🎓 Academic & Portfolio Use

This project is ideal as a portfolio piece for:

Equity research & macro strategy roles

Quant & data science internships

Fintech / trading interviews

It demonstrates:

Quant research (factors, signals, backtests)

Macro-quant integration (macro regimes → allocation)

Software engineering discipline (layered architecture, tests, config, Docker)

Realistic JPMorgan-style CIO dashboard thinking

To cite:

@software{jpm_europe_thesis_monitor_2025,
  author = {V.L. Siddarth},
  title  = {JPMorgan European Equity Thesis Monitor},
  year   = {2025},
  url    = {https://github.com/VLSiddarth/JPMorganChase}
}

🤝 Contributing

Contributions welcome:

Fork the repo

Create a feature branch:
git checkout -b feature/amazing-feature

Commit your changes:
git commit -m "Add amazing feature"

Push and open a Pull Request

Areas to help:

New signal engines (quality, low-vol, size)

Better macro proxies / EU-specific data

Factor models (Fama-French Europe, custom factors)

Sentiment module (FinBERT integration)

Tests & CI (GitHub Actions)

📝 License & Disclaimer

This project is licensed under the MIT License. See LICENSE
.

⚠️ Disclaimer

This is for educational and informational purposes only.

Not investment advice.

Not affiliated with JPMorgan Chase & Co.

Past performance is not indicative of future results.

Always do your own research and consult a licensed financial advisor.

## 📧 Contact

**Your Name**
- GitHub: [@VLSiddarth](https://github.com/VLSiddarth)
- LinkedIn: [V.L.Siddarth](https://www.linkedin.com/in/v-l-siddarth-2147b9250/)
- Email: vlsiddarth7@gmail.com

---

**⭐ If this project helps you, consider giving it a star and sharing it with other equity research students & quants!**

Built with ❤️ for equity research students worldwide# JPMorgan