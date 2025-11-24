# 📊 JPMorgan European Equity Thesis Monitor

> A real-time investment dashboard validating JPMorgan's 2025 European equity overweight thesis

[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

This dashboard provides institutional-grade monitoring and analysis of JPMorgan's 2025 European equity investment thesis. It combines traditional fundamental analysis with machine learning, quantitative backtesting, and alternative data to answer three critical questions:

1. **Is the thesis working?** (CIO View)
2. **Where is it working?** (PM View)
3. **Why is it working (or not)?** (Strategist View)

## ✨ Key Features

### 📈 Real-Time Monitoring
- Live tracking of STOXX 600 vs S&P 500 performance
- 5 key performance indicators (KPIs) with target thresholds
- Sector and thematic basket performance analysis

### 🤖 AI-Powered Insights
- **FinBERT sentiment analysis** of European economic news
- Natural language processing for market sentiment scoring
- Real-time news aggregation and classification

### 🎯 Trade Signal Generation
- Automated buy/sell/hold signals based on multiple factors
- Risk alerts for EU fragmentation and tariff risks
- Conviction-weighted recommendations with timeframes

### 📊 Quantitative Backtesting
- Historical performance validation (2020-present)
- Strategy vs benchmark comparison
- Sharpe ratio, drawdown, and win rate analysis

### 🔔 Alert System
- Email notifications for critical signals
- Customizable alert thresholds
- HTML-formatted detailed reports

### 📉 Factor Analysis
- Exposure analysis across 5 key factors (Value, Momentum, Quality, Size, Low Vol)
- Portfolio vs benchmark comparison
- Factor tilts visualization

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.11+
pip (Python package manager)
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/jpm-dashboard.git
cd jpm-dashboard
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure API keys:**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

4. **Run the dashboard:**
```bash
streamlit run app.py
```

5. **Open in browser:**
```
http://localhost:8501
```

## 🔑 API Keys Setup

### Required (Free)

1. **FRED API** (Macro data)
   - Get key: https://fred.stlouisfed.org/docs/api/api_key.html
   - Add to `.env`: `FRED_API_KEY=your_key_here`

2. **NewsAPI** (News sentiment)
   - Get key: https://newsapi.org/register
   - Free tier: 100 requests/day
   - Add to `.env`: `NEWSAPI_KEY=your_key_here`

### Optional (For email alerts)

3. **Gmail SMTP** (Email alerts)
   - Enable 2FA on Gmail
   - Generate App Password
   - Add to `.env`:
```
     SMTP_EMAIL=your@gmail.com
     SMTP_PASSWORD=your_app_password
```

## 📊 Dashboard Views

### 1. 📈 CIO View (Thesis-at-a-Glance)
**Answer:** Is the JPM thesis working?

- Europe vs US relative performance (3-month rolling)
- Eurozone 2026 EPS growth consensus
- EU-US valuation gap (forward P/E)
- EU fragmentation risk gauge (FR-DE spread)
- Eurozone credit impulse

### 2. 💼 PM View (Sector & Thematic)
**Answer:** Where in Europe is the thesis working?

- STOXX 600 sector performance treemap
- Thematic basket performance:
  - German Fiscal Play (Siemens, Schneider, Vinci, etc.)
  - EU Defense (Rheinmetall, BAE Systems, Thales, Leonardo)
  - GRANOLAS (11 mega-cap stocks)
  - EU Banks (Unicredit, Santander, BBVA, BNP)

### 3. 🌍 Strategist View (Macro & Policy)
**Answer:** Why is the thesis working (or not)?

- German GDP forecasts (2026)
- German IFO Business Climate Index
- China Caixin Manufacturing PMI
- U.S. Tariff Risk Tracker
- EU fragmentation risk analysis

### 4. 🎯 Live Trade Signals
- Real-time buy/sell/hold recommendations
- Signal conviction levels (High/Medium/Low)
- Target allocations and timeframes
- Historical signal tracking (coming soon)

### 5. 📊 Backtest Performance
- Strategy performance since 2020
- Equity curve vs buy-and-hold
- Risk-adjusted metrics
- Drawdown analysis

### 6. 📰 News Sentiment
- AI-powered sentiment analysis
- Recent article breakdown
- Sentiment timeline visualization
- Source attribution

### 7. ⚙️ Settings & Alerts
- Email alert configuration
- API key management
- Data export (JSON/CSV/PDF)
- Dashboard customization

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit 1.31 | Interactive dashboard |
| **Visualization** | Plotly 5.18 | Dynamic charts & graphs |
| **Market Data** | yfinance 0.2.35 | Stock prices & fundamentals |
| **Macro Data** | FRED API | Economic indicators |
| **News** | NewsAPI | Real-time news articles |
| **AI/ML** | FinBERT (Transformers) | Sentiment analysis |
| **Backtesting** | Custom engine | Strategy validation |
| **Alerts** | SMTP (Gmail) | Email notifications |

## 📁 Project Structure
```
JPMorganChase/
│
├── app.py                      # Main Streamlit dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
│
├── data/                       # Cached data files
│   └── dashboard_cache.json
│
├── modules/                    # Core functionality
│   ├── data_loader.py         # Fetch market & macro data
│   ├── signal_generator.py    # Generate trade signals
│   ├── backtest_engine.py     # Backtest strategies
│   ├── sentiment_analyzer.py  # AI sentiment analysis
│   ├── factor_analysis.py     # Factor exposure calculation
│   └── alerts.py              # Email alert system
│
├── utils/                      # Helper functions
│   ├── __init__.py
│   └── helpers.py             # Utility functions
│
└── assets/                     # Static files (optional)
    └── custom.css
```

## 🎓 Academic Use

This project was developed as part of equity research coursework and demonstrates:

- **Quantitative Analysis**: Factor models, backtesting, performance attribution
- **Data Science**: API integration, data pipeline, caching strategies
- **Machine Learning**: NLP sentiment analysis using FinBERT
- **Software Engineering**: Modular design, clean code, documentation
- **Financial Theory**: Portfolio construction, risk management, fundamental analysis

### Citing This Work
```bibtex
@software{jpm_dashboard_2025,
  author = {Your Name},
  title = {JPMorgan European Equity Thesis Monitor},
  year = {2025},
  url = {https://github.com/yourusername/jpm-dashboard}
}
```

## 📈 Usage Examples

### Fetch Latest Data
```python
from modules.data_loader import DataLoader

loader = DataLoader()
data = loader.fetch_all_data()
print(f"Relative Performance: {data['indices']['relative_performance']:.2f}%")
```

### Generate Trade Signals
```python
from modules.signal_generator import SignalGenerator

generator = SignalGenerator()
signals = generator.generate_signals(data)

for signal in signals:
    print(f"{signal['title']}: {signal['action']}")
```

### Run Backtest
```python
from modules.backtest_engine import BacktestEngine

engine = BacktestEngine(start_date='2020-01-01')
results = engine.run_backtest()
print(engine.generate_summary())
```

### Analyze Sentiment
```python
from modules.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
summary = analyzer.get_sentiment_summary()
print(f"Sentiment Score: {summary['score']:.1f}/100")
```

## 🚀 Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add API keys in "Secrets" section
5. Deploy!

Your dashboard will be live at: `https://username-jpm-dashboard.streamlit.app`

### Option 2: Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT" > Procfile

# Deploy
heroku create jpm-dashboard
git push heroku main
```

### Option 3: Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```
```bash
docker build -t jpm-dashboard .
docker run -p 8501:8501 jpm-dashboard
```

## 🔧 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
pip install --upgrade -r requirements.txt
```

**2. FRED API Errors**
- Check your API key in `.env`
- Verify key is active at fred.stlouisfed.org
- Free tier: 120 requests/minute

**3. Yahoo Finance Data Missing**
- Some European tickers need exchange suffix (e.g., `SIE.DE`)
- Try adding `.DE`, `.PA`, `.L`, `.MI` suffixes
- Use ETFs as proxies if direct access fails

**4. FinBERT Model Download**
- First run downloads ~400MB model
- Ensure stable internet connection
- Model caches in `~/.cache/huggingface/`

**5. Email Alerts Not Working**
- Use Gmail App Password (not regular password)
- Enable "Less secure app access" if needed
- Check SMTP settings in `.env`

## 📚 Documentation

- [Streamlit Docs](https://docs.streamlit.io)
- [Plotly Python](https://plotly.com/python/)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [FRED API](https://fred.stlouisfed.org/docs/api/)
- [FinBERT Paper](https://arxiv.org/abs/1908.10063)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**IMPORTANT:** This dashboard is for **educational and informational purposes only**. 

- Not investment advice
- Not affiliated with JPMorgan Chase & Co.
- Past performance does not guarantee future results
- Always do your own research
- Consult a licensed financial advisor before investing

## 🙏 Acknowledgments

- JPMorgan European Equity Research Team (thesis inspiration)
- Streamlit team for amazing framework
- FinBERT authors for sentiment model
- Open-source community

## 📧 Contact

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [your-profile](https://linkedin.com/in/yourprofile)
- Email: vlsiddarth7@gmail.com

---

**⭐ If this project helped you, please consider giving it a star!**

Built with ❤️ for equity research students worldwide# JPMorgan
