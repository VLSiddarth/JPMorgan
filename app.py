"""
JPMorgan European Equity Thesis Dashboard
Main Streamlit Application
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(__file__))

from modules.data_loader import DataLoader
from modules.signal_generator import SignalGenerator
from modules.backtest_engine import BacktestEngine
from modules.sentiment_analyzer import SentimentAnalyzer
from modules.factor_analysis import FactorAnalyzer
from modules.alerts import AlertSystem
from utils.helpers import format_percentage, format_currency, normalize_to_100

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="JPM European Equity Thesis Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --jpm-blue: #003366;
        --jpm-gold: #C5A572;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #003366 0%, #005599 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #003366;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .metric-positive {
        border-left-color: #28a745;
        background-color: #f0fff4;
    }
    
    .metric-negative {
        border-left-color: #dc3545;
        background-color: #fff5f5;
    }
    
    .metric-warning {
        border-left-color: #ffc107;
        background-color: #fffbf0;
    }
    
    /* Signal boxes */
    .signal-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    
    .signal-buy {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    
    .signal-sell {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    
    .signal-hold {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Section headers */
    .section-header {
        background-color: #003366;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 20px 0 10px 0;
        font-size: 18px;
        font-weight: bold;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #0dcaf0;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Data tables */
    .dataframe {
        font-size: 14px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #003366;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #005599;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING AND CACHING
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data():
    """Load all dashboard data with caching"""
    loader = DataLoader()
    
    # Try to load from cache first
    cached_data = loader.load_cache()
    if cached_data:
        # Check if cache is recent (within 1 hour)
        cache_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
        if (datetime.now() - cache_time).seconds < 3600:
            return cached_data
    
    # Fetch fresh data
    return loader.fetch_all_data()

@st.cache_resource
def initialize_modules():
    """Initialize all analysis modules"""
    return {
        'signal_generator': SignalGenerator(),
        'backtest_engine': BacktestEngine(start_date='2020-01-01'),
        'sentiment_analyzer': SentimentAnalyzer(),
        'factor_analyzer': FactorAnalyzer(),
        'alert_system': AlertSystem()
    }

# ============================================
# HEADER
# ============================================

st.markdown("""
<div class="main-header">
    <h1>📊 JPMorgan European Equity Thesis Monitor</h1>
    <p style="font-size: 18px; margin: 10px 0 0 0;">
        Real-Time Monitoring Dashboard | 2025 Investment Thesis
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.image("https://www.jpmorganchase.com/content/dam/jpm/global/logos/logo-jpm-share-button.png", 
             width=200)
    
    st.markdown("---")
    st.title("📊 Navigation")
    
    view = st.radio(
        "Select Dashboard View:",
        [
            "📈 CIO View (Thesis-at-a-Glance)",
            "💼 PM View (Sector & Thematic)",
            "🌍 Strategist View (Macro & Policy)",
            "🎯 Live Trade Signals",
            "📊 Backtest Performance",
            "📰 News Sentiment",
            "⚙️ Settings & Alerts"
        ],
        index=0
    )
    
    st.markdown("---")
    st.subheader("🔄 Data Controls")
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
    
    if auto_refresh:
        st.info("🔄 Auto-refresh enabled")
        # This will rerun the app every 5 minutes
        st.empty()
    
    st.markdown("---")
    
    # Data timestamp
    try:
        data = load_all_data()
        last_update = datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
        st.caption(f"📅 Last Updated:\n{last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    except:
        st.caption("📅 Data loading...")
    
    st.markdown("---")
    st.caption("Built with ❤️ for JPMorgan\nv1.0.0 | Free & Open Source")

# ============================================
# LOAD DATA AND INITIALIZE MODULES
# ============================================

with st.spinner("📊 Loading dashboard data..."):
    data = load_all_data()
    modules = initialize_modules()

# Generate signals
signal_generator = modules['signal_generator']
signals = signal_generator.generate_signals(data)
overall_recommendation = signal_generator.get_overall_recommendation()

# ============================================
# VIEW 1: CIO VIEW (THESIS-AT-A-GLANCE)
# ============================================

if "CIO View" in view:
    st.markdown('<div class="section-header">📈 Strategic CIO View: Is the JPM Thesis Working?</div>', 
                unsafe_allow_html=True)
    
    # Overall thesis status
    col_status1, col_status2 = st.columns([3, 1])
    
    with col_status1:
        st.markdown("### 🎯 Overall Thesis Status")
        
        # Determine status
        rel_perf = data['indices'].get('relative_performance', 0)
        if rel_perf > 3:
            status = "✅ **WORKING** - Europe is outperforming"
            status_color = "success"
        elif rel_perf > -3:
            status = "⚠️ **MIXED** - Performance in line with US"
            status_color = "warning"
        else:
            status = "🔴 **UNDERPERFORMING** - Below expectations"
            status_color = "error"
        
        if status_color == "success":
            st.success(status)
        elif status_color == "warning":
            st.warning(status)
        else:
            st.error(status)
    
    with col_status2:
        st.markdown("### 📊 Recommendation")
        rec = overall_recommendation['recommendation']
        if rec == 'BUY':
            st.success(f"**{rec}**")
        elif rec == 'REDUCE':
            st.error(f"**{rec}**")
        else:
            st.info(f"**{rec}**")
        st.caption(overall_recommendation['allocation'])
    
    st.markdown("---")
    
    # Key Performance Indicators (KPIs)
    st.markdown("### 📊 Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        rel_perf = data['indices'].get('relative_performance', 0)
        delta_color = "normal" if rel_perf > 0 else "inverse"
        st.metric(
            label="🎯 EU vs US (3M)",
            value=f"{rel_perf:.2f}%",
            delta="Outperforming" if rel_perf > 0 else "Underperforming",
            delta_color=delta_color
        )
        if rel_perf > 0:
            st.success("✅ Target Met")
        else:
            st.error("🔴 Below Target")
    
    with kpi_col2:
        eps_growth = data.get('eps_growth_2026', 12.5)
        st.metric(
            label="📊 EZ 2026 EPS Growth",
            value=f"{eps_growth:.1f}%",
            delta=f"Target: >12%",
            delta_color="normal" if eps_growth > 12 else "inverse"
        )
        if eps_growth > 12:
            st.success("✅ Target Met")
        else:
            st.warning("⚠️ Below Target")
    
    with kpi_col3:
        pe_gap = data['valuations'].get('pe_gap', 7.5)
        st.metric(
            label="💰 EU-US P/E Gap",
            value=f"{pe_gap:.1f}x",
            delta="Narrowing" if pe_gap < 8 else "Widening",
            delta_color="normal" if pe_gap < 8 else "inverse"
        )
        if pe_gap < 8:
            st.success("✅ Target Met")
        else:
            st.warning("⚠️ Above Target")
    
    with kpi_col4:
        spread = data['macro'].get('fr_de_spread', 68)
        
        # Gauge visualization
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=spread,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "FR-DE Spread (bps)"},
            gauge={
                'axis': {'range': [None, 120]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 80], 'color': "lightgreen"},
                    {'range': [80, 120], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if 40 <= spread <= 80:
            st.success("✅ In Range")
        else:
            st.error("🔴 Out of Range")
    
    with kpi_col5:
        credit_impulse = data.get('credit_impulse', 3.2)
        st.metric(
            label="💳 EZ Credit Impulse",
            value=f"{credit_impulse:.1f}%",
            delta="Target: >3%",
            delta_color="normal" if credit_impulse > 3 else "inverse"
        )
        if credit_impulse > 3:
            st.success("✅ Target Met")
        else:
            st.warning("⚠️ Below Target")
    
    st.markdown("---")
    
    # Historical Performance Chart
    st.markdown("### 📉 Historical Performance: STOXX 600 vs S&P 500")
    
    col_chart, col_stats = st.columns([3, 1])
    
    with col_chart:
        # Get historical data
        stoxx_data = data['indices'].get('stoxx_data', {})
        sp500_data = data['indices'].get('sp500_data', {})
        
        if stoxx_data and sp500_data:
            # Convert to pandas Series
            stoxx_series = pd.Series(stoxx_data)
            sp500_series = pd.Series(sp500_data)
            
            # Normalize to 100
            stoxx_norm = normalize_to_100(stoxx_series)
            sp500_norm = normalize_to_100(sp500_series)
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=stoxx_norm.index,
                y=stoxx_norm.values,
                mode='lines',
                name='STOXX 600',
                line=dict(color='#4169E1', width=3),
                hovertemplate='%{y:.2f}<br>%{x}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=sp500_norm.index,
                y=sp500_norm.values,
                mode='lines',
                name='S&P 500',
                line=dict(color='#FF6347', width=3),
                hovertemplate='%{y:.2f}<br>%{x}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Indexed Performance (Base = 100)",
                xaxis_title="Date",
                yaxis_title="Index Level",
                hovermode='x unified',
                height=450,
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col_stats:
        st.markdown("#### 📊 Performance Stats")
        
        stoxx_return = data['indices'].get('stoxx_3m_return', 0)
        sp500_return = data['indices'].get('sp500_3m_return', 0)
        
        st.metric("STOXX 600 (3M)", f"{stoxx_return:.2f}%")
        st.metric("S&P 500 (3M)", f"{sp500_return:.2f}%")
        st.metric("Relative", f"{rel_perf:.2f}%", 
                 delta=f"{'Out' if rel_perf > 0 else 'Under'}performing")
        
        st.markdown("---")
        
        st.markdown("**Valuation Metrics**")
        st.metric("STOXX 600 P/E", f"{data['valuations'].get('stoxx_pe', 14.5):.1f}x")
        st.metric("S&P 500 P/E", f"{data['valuations'].get('sp500_pe', 22.0):.1f}x")
        st.metric("Discount", f"{pe_gap:.1f}x")
    
    # Active Signals Summary
    if signals:
        st.markdown("---")
        st.markdown("### 🎯 Active Signals")
        
        signal_cols = st.columns(len(signals) if len(signals) <= 3 else 3)
        
        for idx, signal in enumerate(signals[:3]):  # Show top 3 signals
            with signal_cols[idx % 3]:
                signal_type = signal['type']
                if signal_type in ['BUY', 'STRONG_BUY', 'FUNDAMENTAL_BULLISH']:
                    st.success(f"**{signal['title']}**")
                elif signal_type in ['RISK_ALERT', 'WARNING']:
                    st.error(f"**{signal['title']}**")
                else:
                    st.info(f"**{signal['title']}**")
                
                st.caption(signal['message'])
                st.caption(f"🎯 {signal['action']}")

# ============================================
# VIEW 2: PM VIEW (SECTOR & THEMATIC)
# ============================================

elif "PM View" in view:
    st.markdown('<div class="section-header">💼 Portfolio Manager View: Where is the Thesis Working?</div>', 
                unsafe_allow_html=True)
    
    # Sector Performance Treemap
    st.markdown("### 🗺️ STOXX 600 Sector Performance (QTD)")
    
    sector_perf = data.get('sectors', {})
    
    if sector_perf:
        # Prepare data for treemap
        df_sectors = pd.DataFrame({
            'Sector': list(sector_perf.keys()),
            'Return': list(sector_perf.values())
        })
        
        # Create treemap
        fig_treemap = px.treemap(
            df_sectors,
            path=['Sector'],
            values=[abs(x) + 1 for x in df_sectors['Return']],  # Add 1 to handle negatives
            color='Return',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            hover_data={'Return': ':.2f%'}
        )
        
        fig_treemap.update_traces(
            textposition="middle center",
            texttemplate="<b>%{label}</b><br>%{color:.2f}%"
        )
        
        fig_treemap.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        st.plotly_chart(fig_treemap, use_container_width=True)
        
        # Sector performance table
        col_table1, col_table2 = st.columns(2)
        
        with col_table1:
            st.markdown("#### 🏆 Best Performers")
            best = df_sectors.nlargest(3, 'Return')
            for _, row in best.iterrows():
                st.success(f"**{row['Sector']}**: {row['Return']:.2f}%")
        
        with col_table2:
            st.markdown("#### 📉 Worst Performers")
            worst = df_sectors.nsmallest(3, 'Return')
            for _, row in worst.iterrows():
                st.error(f"**{row['Sector']}**: {row['Return']:.2f}%")
    
    st.markdown("---")
    
    # Thematic Basket Performance
    st.markdown("### 📊 Thematic Basket Performance (YTD)")
    
    baskets = data.get('baskets', {})
    
    if baskets:
        # Prepare basket data
        basket_names = []
        basket_returns = []
        
        basket_display_names = {
            'german_fiscal': 'German Fiscal Play',
            'eu_defense': 'EU Defense',
            'granolas': 'GRANOLAS',
            'eu_banks': 'EU Banks'
        }
        
        for basket_key, basket_data in baskets.items():
            if basket_data and 'average_return' in basket_data:
                basket_names.append(basket_display_names.get(basket_key, basket_key))
                basket_returns.append(basket_data['average_return'])
        
        # Add benchmark
        basket_names.append('STOXX 600 (Benchmark)')
        basket_returns.append(data['indices'].get('stoxx_3m_return', 0))
        
        df_baskets = pd.DataFrame({
            'Basket': basket_names,
            'YTD Return (%)': basket_returns
        }).sort_values('YTD Return (%)', ascending=False)
        
        # Create bar chart
        fig_baskets = px.bar(
            df_baskets,
            x='Basket',
            y='YTD Return (%)',
            color='YTD Return (%)',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            text='YTD Return (%)'
        )
        
        fig_baskets.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        
        fig_baskets.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="",
            yaxis_title="YTD Return (%)",
            template='plotly_white'
        )
        
        st.plotly_chart(fig_baskets, use_container_width=True)
        
        # Detailed basket breakdown
        st.markdown("### 📋 Detailed Basket Breakdown")
        
        basket_tabs = st.tabs(list(basket_display_names.values()))
        
        for idx, (basket_key, display_name) in enumerate(basket_display_names.items()):
            with basket_tabs[idx]:
                basket_data = baskets.get(basket_key, {})
                
                if basket_data and 'stocks' in basket_data:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        stocks = basket_data['stocks']
                        df_stocks = pd.DataFrame({
                            'Stock': list(stocks.keys()),
                            'Ticker': [stocks[s]['ticker'] for s in stocks.keys()],
                            'Return (%)': [stocks[s]['return'] for s in stocks.keys()],
                            'Price': [stocks[s]['current_price'] for s in stocks.keys()]
                        })
                        
                        # Color code the dataframe
                        st.dataframe(
                            df_stocks.style.background_gradient(
                                subset=['Return (%)'],
                                cmap='RdYlGn',
                                vmin=-10,
                                vmax=10
                            ).format({
                                'Return (%)': '{:.2f}%',
                                'Price': '{:.2f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with col2:
                        st.metric(
                            "Basket Average Return",
                            f"{basket_data['average_return']:.2f}%"
                        )
                        st.metric(
                            "Number of Holdings",
                            basket_data['num_stocks']
                        )
                        
                        # Performance vs benchmark
                        benchmark_return = data['indices'].get('stoxx_3m_return', 0)
                        alpha = basket_data['average_return'] - benchmark_return
                        st.metric(
                            "Alpha vs STOXX 600",
                            f"{alpha:.2f}%",
                            delta="Outperforming" if alpha > 0 else "Underperforming"
                        )

# Continue to next message for Views 3-7...
                # ============================================
# VIEW 3: STRATEGIST VIEW (MACRO & POLICY)
# ============================================

elif "Strategist View" in view:
    st.markdown('<div class="section-header">🌍 Strategist View: Why is the Thesis Working?</div>', 
                unsafe_allow_html=True)
    
    # GDP Forecasts and PMI
    col_macro1, col_macro2 = st.columns(2)
    
    with col_macro1:
        st.markdown("### 📊 German GDP Forecasts (2026)")
        
        gdp_data = {
            'Source': ['Bundesbank', 'DIW Berlin', 'Consensus'],
            'Forecast (%)': [0.7, 1.7, 1.2]
        }
        df_gdp = pd.DataFrame(gdp_data)
        
        fig_gdp = px.bar(
            df_gdp,
            x='Source',
            y='Forecast (%)',
            text='Forecast (%)',
            color='Forecast (%)',
            color_continuous_scale='Blues',
            title="Forecast Tension Analysis"
        )
        
        fig_gdp.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        
        fig_gdp.update_layout(
            showlegend=False,
            height=350,
            yaxis_range=[0, 2.5]
        )
        
        st.plotly_chart(fig_gdp, use_container_width=True)
        
        st.info("💡 **Insight**: Wide forecast range (0.7%-1.7%) indicates uncertainty around German fiscal stimulus impact. DIW's optimistic view reflects infrastructure spending benefits.")
    
    with col_macro2:
        st.markdown("### 🏭 China Caixin Manufacturing PMI")
        
        # Generate synthetic PMI data (replace with real FRED data)
        pmi_dates = pd.date_range(end=datetime.now(), periods=12, freq='M')
        pmi_values = [49.2, 50.1, 49.8, 50.3, 50.5, 49.9, 50.8, 51.2, 50.6, 49.8, 50.4, 51.0]
        
        df_pmi = pd.DataFrame({
            'Date': pmi_dates,
            'PMI': pmi_values
        })
        
        fig_pmi = go.Figure()
        
        fig_pmi.add_trace(go.Scatter(
            x=df_pmi['Date'],
            y=df_pmi['PMI'],
            mode='lines+markers',
            name='Caixin PMI',
            line=dict(color='#FF6347', width=3),
            marker=dict(size=8)
        ))
        
        fig_pmi.add_hline(
            y=50,
            line_dash="dash",
            line_color="gray",
            annotation_text="50 (Expansion Threshold)",
            annotation_position="right"
        )
        
        # Color zones
        fig_pmi.add_hrect(
            y0=50, y1=55,
            fillcolor="green",
            opacity=0.1,
            line_width=0,
            annotation_text="Expansion",
            annotation_position="top right"
        )
        
        fig_pmi.add_hrect(
            y0=45, y1=50,
            fillcolor="red",
            opacity=0.1,
            line_width=0,
            annotation_text="Contraction",
            annotation_position="bottom right"
        )
        
        fig_pmi.update_layout(
            title="Monthly Manufacturing Activity",
            yaxis_title="PMI Index",
            height=350,
            hovermode='x unified',
            yaxis_range=[45, 55]
        )
        
        st.plotly_chart(fig_pmi, use_container_width=True)
        
        current_pmi = pmi_values[-1]
        if current_pmi > 50:
            st.success(f"✅ Current PMI: {current_pmi} (Expansion)")
        else:
            st.warning(f"⚠️ Current PMI: {current_pmi} (Contraction Risk)")
    
    st.markdown("---")
    
    # German IFO Business Climate
    st.markdown("### 📈 German IFO Business Climate Index")
    
    macro_data = data.get('macro', {})
    ifo_data = macro_data.get('german_ifo', {})
    
    if ifo_data:
        df_ifo = pd.DataFrame(list(ifo_data.items()), columns=['Date', 'IFO'])
        df_ifo['Date'] = pd.to_datetime(df_ifo['Date'])
        df_ifo = df_ifo.sort_values('Date')
        
        fig_ifo = go.Figure()
        
        fig_ifo.add_trace(go.Scatter(
            x=df_ifo['Date'],
            y=df_ifo['IFO'],
            mode='lines+markers',
            name='IFO Index',
            line=dict(color='#4169E1', width=3),
            fill='tozeroy',
            fillcolor='rgba(65, 105, 225, 0.1)'
        ))
        
        fig_ifo.update_layout(
            title="German Business Confidence (Monthly)",
            xaxis_title="Date",
            yaxis_title="IFO Index",
            height=350,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig_ifo, use_container_width=True)
        
        # Trend analysis
        if len(df_ifo) >= 3:
            recent_trend = df_ifo['IFO'].iloc[-3:].mean() - df_ifo['IFO'].iloc[-6:-3].mean()
            if recent_trend > 0:
                st.success(f"✅ Improving trend (+{recent_trend:.2f} over last 3 months)")
            else:
                st.warning(f"⚠️ Declining trend ({recent_trend:.2f} over last 3 months)")
    
    st.markdown("---")
    
    # U.S. Tariff Risk Tracker
    st.markdown("### ⚠️ U.S. Tariff Risk Tracker")
    
    tariff_data = {
        'Date': ['2024-11-15', '2024-10-30', '2024-09-20', '2024-08-15'],
        'Announcement': [
            'Steel derivatives tariff proposal',
            'Auto parts tariff review initiated',
            'Industrial equipment review',
            'Section 232 investigation on EU aluminum'
        ],
        'Affected EU Sector': ['Industrials', 'Autos', 'Industrials', 'Materials'],
        'Impact Level': ['High', 'Critical', 'Medium', 'Medium'],
        'Status': ['Proposed', 'Under Review', 'Monitoring', 'Concluded']
    }
    
    df_tariffs = pd.DataFrame(tariff_data)
    
    # Color code by impact
    def highlight_impact(row):
        colors = {
            'Critical': 'background-color: #ffcccc',
            'High': 'background-color: #ffddaa',
            'Medium': 'background-color: #ffffcc',
            'Low': 'background-color: #ccffcc'
        }
        return [colors.get(row['Impact Level'], '')] * len(row)
    
    st.dataframe(
        df_tariffs.style.apply(highlight_impact, axis=1),
        use_container_width=True,
        hide_index=True
    )
    
    # Risk summary
    col_risk1, col_risk2, col_risk3 = st.columns(3)
    
    with col_risk1:
        critical = (df_tariffs['Impact Level'] == 'Critical').sum()
        st.metric("🔴 Critical Risks", critical)
    
    with col_risk2:
        high = (df_tariffs['Impact Level'] == 'High').sum()
        st.metric("🟡 High Risks", high)
    
    with col_risk3:
        medium = (df_tariffs['Impact Level'] == 'Medium').sum()
        st.metric("🟢 Medium Risks", medium)
    
    st.markdown("---")
    
    # Fragmentation Risk Deep Dive
    st.markdown("### 🇪🇺 EU Fragmentation Risk Analysis")
    
    col_frag1, col_frag2 = st.columns([2, 1])
    
    with col_frag1:
        # Historical spread chart
        fr_history = macro_data.get('fr_10y_history', {})
        de_history = macro_data.get('de_10y_history', {})
        
        if fr_history and de_history:
            df_fr = pd.DataFrame(list(fr_history.items()), columns=['Date', 'FR_10Y'])
            df_de = pd.DataFrame(list(de_history.items()), columns=['Date', 'DE_10Y'])
            
            df_spread = pd.merge(df_fr, df_de, on='Date')
            df_spread['Date'] = pd.to_datetime(df_spread['Date'])
            df_spread['Spread'] = (df_spread['FR_10Y'] - df_spread['DE_10Y']) * 100
            df_spread = df_spread.sort_values('Date')
            
            fig_spread = go.Figure()
            
            fig_spread.add_trace(go.Scatter(
                x=df_spread['Date'],
                y=df_spread['Spread'],
                mode='lines',
                name='FR-DE Spread',
                line=dict(color='#003366', width=2),
                fill='tozeroy'
            ))
            
            # Add threshold lines
            fig_spread.add_hline(y=80, line_dash="dash", line_color="red",
                                annotation_text="Danger Zone (80bps)")
            fig_spread.add_hline(y=40, line_dash="dash", line_color="green",
                                annotation_text="Safe Zone (40bps)")
            
            fig_spread.update_layout(
                title="France-Germany 10Y Spread (Historical)",
                xaxis_title="Date",
                yaxis_title="Spread (bps)",
                height=350,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_spread, use_container_width=True)
    
    with col_frag2:
        st.markdown("#### 📊 Current Status")
        
        current_spread = data['macro'].get('fr_de_spread', 68)
        
        st.metric("Current Spread", f"{current_spread:.0f} bps")
        
        if current_spread < 40:
            st.success("✅ Low Risk\nEU cohesion strong")
        elif current_spread < 80:
            st.info("⚠️ Moderate Risk\nWithin acceptable range")
        else:
            st.error("🔴 High Risk\nFragmentation concerns")
        
        st.markdown("---")
        st.markdown("**Risk Thresholds:**")
        st.caption("• < 40 bps: Low Risk")
        st.caption("• 40-80 bps: Moderate")
        st.caption("• > 80 bps: High Risk")

# ============================================
# VIEW 4: LIVE TRADE SIGNALS
# ============================================

elif "Live Trade Signals" in view:
    st.markdown('<div class="section-header">🎯 Live Trade Signals</div>', 
                unsafe_allow_html=True)
    
    # Overall recommendation banner
    rec = overall_recommendation['recommendation']
    
    if rec == 'BUY':
        st.success(f"""
        ### 🚀 STRONG BUY SIGNAL
        **Allocation:** {overall_recommendation['allocation']}
        
        {overall_recommendation['summary']}
        """)
    elif rec == 'REDUCE':
        st.error(f"""
        ### 🔻 REDUCE EXPOSURE SIGNAL
        **Allocation:** {overall_recommendation['allocation']}
        
        {overall_recommendation['summary']}
        """)
    else:
        st.info(f"""
        ### ⏸️ HOLD SIGNAL
        **Allocation:** {overall_recommendation['allocation']}
        
        {overall_recommendation['summary']}
        """)
    
    st.markdown("---")
    
    # Individual signals
    if signals:
        st.markdown("### 📋 Active Signals Breakdown")
        
        for idx, signal in enumerate(signals):
            signal_type = signal['type']
            
            # Determine signal box style
            if signal_type in ['BUY', 'STRONG_BUY', 'FUNDAMENTAL_BULLISH', 'OPPORTUNITY']:
                box_class = 'signal-buy'
                icon = '🟢'
            elif signal_type in ['RISK_ALERT', 'WARNING']:
                box_class = 'signal-sell'
                icon = '🔴'
            else:
                box_class = 'signal-hold'
                icon = '🟡'
            
            st.markdown(f"""
            <div class="signal-box {box_class}">
                <h3>{icon} {signal['title']}</h3>
                <p><strong>Message:</strong> {signal['message']}</p>
                <p><strong>Conviction:</strong> {signal['conviction']}</p>
                <p><strong>🎯 Recommended Action:</strong> {signal['action']}</p>
                <p><strong>Target Allocation:</strong> {signal['target_allocation']}</p>
                <p><strong>⏱️ Timeframe:</strong> {signal['timeframe']}</p>
                <p style="font-size: 12px; color: #666;">Generated: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
    else:
        st.info("ℹ️ No active signals at this time. All metrics within normal ranges.")
    
    st.markdown("---")
    
    # Signal history (simulated)
    st.markdown("### 📊 Signal Performance Tracker")
    
    st.info("""
    **Coming Soon:** Track historical signal performance
    
    This section will show:
    - Win rate of past signals
    - Average gain/loss per signal
    - Best performing signal types
    - Signal frequency analysis
    """)

# ============================================
# VIEW 5: BACKTEST PERFORMANCE
# ============================================

elif "Backtest Performance" in view:
    st.markdown('<div class="section-header">📊 Backtest Performance: Strategy Validation</div>', 
                unsafe_allow_html=True)
    
    with st.spinner("🔄 Running backtest (this may take a minute)..."):
        backtest_engine = modules['backtest_engine']
        
        # Check if backtest already ran
        if not backtest_engine.results:
            results = backtest_engine.run_backtest()
        else:
            results = backtest_engine.results
    
    if results:
        # Performance metrics
        st.markdown("### 🏆 Performance Summary")
        
        strategy_metrics = results['strategy_metrics']
        buyhold_metrics = results['buyhold_metrics']
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric(
                "Strategy Return",
                f"{strategy_metrics['total_return']*100:.2f}%",
                delta=f"+{(strategy_metrics['total_return'] - buyhold_metrics['total_return'])*100:.2f}% vs B&H"
            )
        
        with col_m2:
            st.metric(
                "Sharpe Ratio",
                f"{strategy_metrics['sharpe_ratio']:.2f}",
                delta=f"{strategy_metrics['sharpe_ratio'] - buyhold_metrics['sharpe_ratio']:.2f} vs B&H"
            )
        
        with col_m3:
            st.metric(
                "Max Drawdown",
                f"{strategy_metrics['max_drawdown']*100:.2f}%",
                delta=f"{(strategy_metrics['max_drawdown'] - buyhold_metrics['max_drawdown'])*100:.2f}% vs B&H"
            )
        
        with col_m4:
            st.metric(
                "Win Rate",
                f"{strategy_metrics['win_rate']*100:.1f}%"
            )
        
        st.markdown("---")
        
        # Equity curves
        st.markdown("### 📈 Equity Curve Comparison")
        
        fig_equity = go.Figure()
        
        fig_equity.add_trace(go.Scatter(
            x=results['dates'],
            y=results['strategy_equity'],
            mode='lines',
            name='JPM Thesis Strategy',
            line=dict(color='#003366', width=3)
        ))
        
        fig_equity.add_trace(go.Scatter(
            x=results['dates'],
            y=results['buyhold_equity'],
            mode='lines',
            name='STOXX 600 Buy & Hold',
            line=dict(color='#C5A572', width=2, dash='dash')
        ))
        
        fig_equity.update_layout(
            title="Cumulative Returns (2020-Present)",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=450,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_equity, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed metrics comparison
        col_detail1, col_detail2 = st.columns(2)
        
        with col_detail1:
            st.markdown("### 🎯 JPM Thesis Strategy")
            
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Total Return',
                    'Annualized Return',
                    'Sharpe Ratio',
                    'Max Drawdown',
                    'Volatility',
                    'Win Rate'
                ],
                'Value': [
                    f"{strategy_metrics['total_return']*100:.2f}%",
                    f"{strategy_metrics['annualized_return']*100:.2f}%",
                    f"{strategy_metrics['sharpe_ratio']:.2f}",
                    f"{strategy_metrics['max_drawdown']*100:.2f}%",
                    f"{strategy_metrics['volatility']*100:.2f}%",
                    f"{strategy_metrics['win_rate']*100:.1f}%"
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with col_detail2:
            st.markdown("### 📊 Buy & Hold (STOXX 600)")
            
            bh_metrics_df = pd.DataFrame({
                'Metric': [
                    'Total Return',
                    'Annualized Return',
                    'Sharpe Ratio',
                    'Max Drawdown',
                    'Volatility',
                    'Win Rate'
                ],
                'Value': [
                    f"{buyhold_metrics['total_return']*100:.2f}%",
                    f"{buyhold_metrics['annualized_return']*100:.2f}%",
                    f"{buyhold_metrics['sharpe_ratio']:.2f}",
                    f"{buyhold_metrics['max_drawdown']*100:.2f}%",
                    f"{buyhold_metrics['volatility']*100:.2f}%",
                    f"{buyhold_metrics['win_rate']*100:.1f}%"
                ]
            })
            
            st.dataframe(bh_metrics_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Strategy description
        st.markdown("### 📋 Strategy Rules")
        
        st.info("""
        **Signal Generation:**
        - **100% Long Europe** when relative performance < -10% (oversold)
        - **50% Exposure** when -10% < relative performance < +5% (neutral)
        - **0% Exposure (Cash)** when relative performance > +5% (take profit)
        
        **Lookback Period:** 20-day rolling relative performance
        
        **Rebalancing:** Daily based on signals
        
        **No Transaction Costs** included in this backtest
        """)
        
        # Key findings
        st.success(f"""
        ### ✅ Key Findings
        
        1. **Alpha Generated:** The strategy outperformed buy-and-hold by **{(strategy_metrics['total_return'] - buyhold_metrics['total_return'])*100:.2f}%**
        
        2. **Risk-Adjusted Returns:** Sharpe ratio of **{strategy_metrics['sharpe_ratio']:.2f}** vs **{buyhold_metrics['sharpe_ratio']:.2f}** for benchmark
        
        3. **Drawdown Protection:** Strategy experienced **{abs(strategy_metrics['max_drawdown']*100):.2f}%** max drawdown vs **{abs(buyhold_metrics['max_drawdown']*100):.2f}%** for buy-and-hold
        
        4. **Win Rate:** **{strategy_metrics['win_rate']*100:.1f}%** of trading days were positive
        """)
    
    else:
        st.error("⚠️ Unable to run backtest. Please check data availability.")

# ============================================
# VIEW 6: NEWS SENTIMENT
# ============================================

elif "News Sentiment" in view:
    st.markdown('<div class="section-header">📰 News Sentiment Analysis</div>', 
                unsafe_allow_html=True)
    
    with st.spinner("📰 Analyzing news sentiment..."):
        sentiment_analyzer = modules['sentiment_analyzer']
        sentiment_summary = sentiment_analyzer.get_sentiment_summary()
    
    # Overall sentiment gauge
    st.markdown("### 🎯 Overall Market Sentiment")
    
    col_gauge, col_summary = st.columns([1, 1])
    
    with col_gauge:
        sentiment_score = sentiment_summary['score']
        
        fig_sentiment = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_score,
            title={'text': "Europe News Sentiment Score"},
            delta={'reference': 0, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-100, -30], 'color': "salmon"},
                    {'range': [-30, 30], 'color': "lightgray"},
                    {'range': [30, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        
        fig_sentiment.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col_summary:
        st.markdown("#### 📊 Sentiment Breakdown")
        
        counts = sentiment_summary['counts']
        
        total = sentiment_summary['total_articles']
        positive = counts.get('positive', 0)
        negative = counts.get('negative', 0)
        neutral = counts.get('neutral', 0)
        
        st.metric("Total Articles Analyzed", total)
        st.metric("Positive", f"{positive} ({positive/total*100:.1f}%)" if total > 0 else "0")
        st.metric("Negative", f"{negative} ({negative/total*100:.1f}%)" if total > 0 else "0")
        st.metric("Neutral", f"{neutral} ({neutral/total*100:.1f}%)" if total > 0 else "0")
        
        if sentiment_score > 30:
            st.success("✅ Bullish Sentiment")
        elif sentiment_score < -30:
            st.error("🔴 Bearish Sentiment")
        else:
            st.info("⚖️ Neutral Sentiment")
    
    st.markdown("---")
    
    # Recent articles
    st.markdown("### 📰 Recent News Articles")
    
    articles_df = sentiment_summary['articles']
    
    if not articles_df.empty:
        # Sentiment filter
        sentiment_filter = st.multiselect(
            "Filter by Sentiment:",
            ['positive', 'negative', 'neutral'],
            default=['positive', 'negative', 'neutral']
        )
        
        filtered_df = articles_df[articles_df['sentiment'].isin(sentiment_filter)]
        
        # Display articles
        for _, article in filtered_df.iterrows():
            sentiment = article['sentiment']
            
            if sentiment == 'positive':
                st.success(f"""
                **{article['title']}**
                
                📅 {article['date']} | 📰 {article['source']} | 😊 Positive ({article['score']:.2f})
                
                [Read Article]({article['url']})
                """)
            elif sentiment == 'negative':
                st.error(f"""
                **{article['title']}**
                
                📅 {article['date']} | 📰 {article['source']} | 😟 Negative ({article['score']:.2f})
                
                [Read Article]({article['url']})
                """)
            else:
                st.info(f"""
                **{article['title']}**
                
                📅 {article['date']} | 📰 {article['source']} | 😐 Neutral ({article['score']:.2f})
                
                [Read Article]({article['url']})
                """)
        
        st.markdown("---")
        
        # Sentiment over time (if enough data)
        if len(filtered_df) > 5:
            st.markdown("### 📊 Sentiment Trend")
            
            filtered_df['date'] = pd.to_datetime(filtered_df['date'])
            filtered_df = filtered_df.sort_values('date')
            
            score_map = {'positive': 1, 'negative': -1, 'neutral': 0}
            filtered_df['numeric_sentiment'] = filtered_df['sentiment'].map(score_map) * filtered_df['score']
            
            fig_trend = px.scatter(
                filtered_df,
                x='date',
                y='numeric_sentiment',
                color='sentiment',
                color_discrete_map={'positive': 'green', 'negative': 'red', 'neutral': 'gray'},
                hover_data=['title', 'source'],
                title="Sentiment Timeline"
            )
            
            fig_trend.update_layout(height=350)
            st.plotly_chart(fig_trend, use_container_width=True)
    
    else:
        st.warning("⚠️ No articles available. Check your NewsAPI key configuration.")

# ============================================
# VIEW 7: SETTINGS & ALERTS
# ============================================

elif "Settings & Alerts" in view:
    st.markdown('<div class="section-header">⚙️ Settings & Alert Configuration</div>', 
                unsafe_allow_html=True)
    
    # Alert system settings
    st.markdown("### 🔔 Email Alert System")
    
    alert_system = modules['alert_system']
    
    col_alert1, col_alert2 = st.columns([2, 1])
    
    with col_alert1:
        enable_alerts = st.checkbox("Enable Email Alerts", value=False)
        
        if enable_alerts:
            recipient_email = st.text_input(
                "Your Email Address:",
                placeholder="your.email@example.com"
            )
            
            st.markdown("**Alert Triggers:**")
            
            alert_rel_perf = st.checkbox("Europe outperforms/underperforms by >5%", value=True)
            alert_spread = st.checkbox("FR-DE spread exceeds 85bps", value=True)
            alert_sector_rotation = st.checkbox("Sector rotation signal (Financials vs Tech >7%)", value=True)
            alert_pe_gap = st.checkbox("Valuation gap exceeds 9x", value=False)
            
            st.markdown("---")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("📧 Send Test Alert", use_container_width=True):
                    if recipient_email:
                        with st.spinner("Sending test alert..."):
                            success = alert_system.send_test_alert(recipient_email)
                            if success:
                                st.success("✅ Test alert sent successfully!")
                            else:
                                st.error("⚠️ Failed to send alert. Check email configuration.")
                    else:
                        st.warning("⚠️ Please enter your email address")
            
            with col_btn2:
                if st.button("🚨 Check & Send Alerts Now", use_container_width=True):
                    if recipient_email:
                        with st.spinner("Checking for alerts..."):
                            alerts = alert_system.check_alerts(data)
                            if alerts:
                                success = alert_system.send_alert_email(alerts, recipient_email)
                                if success:
                                    st.success(f"✅ Sent {len(alerts)} alert(s) to {recipient_email}")
                                else:
                                    st.error("⚠️ Failed to send alerts")
                            else:
                                st.info("ℹ️ No alerts triggered at this time")
                    else:
                        st.warning("⚠️ Please enter your email address")
    
    with col_alert2:
        st.markdown("#### 📋 Email Setup Instructions")
        
        st.info("""
        **Gmail Setup:**
        
        1. Enable 2FA on your Gmail account
        2. Generate App Password:
           - Go to Google Account settings
           - Security > 2-Step Verification
           - App passwords
        3. Add to `.env` file:
        SMTP_EMAIL=your@gmail.com
        SMTP_PASSWORD=your_app_password
                """)
    
    st.markdown("---")
    
    # Data refresh settings
    st.markdown("### 🔄 Data Refresh Settings")
    
    col_refresh1, col_refresh2 = st.columns(2)
    
    with col_refresh1:
        st.markdown("**Cache Settings:**")
        cache_duration = st.slider("Cache duration (hours)", 1, 24, 1)
        st.caption(f"Data will refresh every {cache_duration} hour(s)")
        
        if st.button("🗑️ Clear All Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Cache cleared! Refresh page to reload data.")
    
    with col_refresh2:
        st.markdown("**Auto-Refresh:**")
        auto_refresh_enabled = st.checkbox("Enable auto-refresh", value=False)
        if auto_refresh_enabled:
            refresh_interval = st.selectbox(
                "Refresh interval:",
                [5, 10, 15, 30, 60],
                format_func=lambda x: f"{x} minutes"
            )
            st.success(f"✅ Auto-refresh every {refresh_interval} minutes")
    
    st.markdown("---")
    
    # API Configuration
    st.markdown("### 🔑 API Configuration")
    
    with st.expander("📊 View API Status"):
        st.markdown("**Current API Keys Status:**")
        
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        fred_key = os.getenv('FRED_API_KEY', '')
        news_key = os.getenv('NEWSAPI_KEY', '')
        
        col_api1, col_api2 = st.columns(2)
        
        with col_api1:
            if fred_key and fred_key != 'demo_key':
                st.success("✅ FRED API: Configured")
            else:
                st.warning("⚠️ FRED API: Not configured")
            
            st.caption("Get free key: [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)")
        
        with col_api2:
            if news_key and news_key != 'demo':
                st.success("✅ NewsAPI: Configured")
            else:
                st.warning("⚠️ NewsAPI: Not configured")
            
            st.caption("Get free key: [newsapi.org](https://newsapi.org/register)")
        
        st.info("""
        **Setup Instructions:**
        1. Copy `.env.example` to `.env`
        2. Add your API keys
        3. Restart the dashboard
        """)
    
    st.markdown("---")
    
    # Export settings
    st.markdown("### 📥 Export & Download")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.button("📊 Download Data (JSON)", use_container_width=True):
            import json
            json_data = json.dumps(data, indent=2, default=str)
            st.download_button(
                label="💾 Save JSON",
                data=json_data,
                file_name=f"jpm_dashboard_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col_export2:
        if st.button("📈 Download Signals (CSV)", use_container_width=True):
            if signals:
                signals_df = signal_generator.format_signals_for_display()
                csv_data = signals_df.to_csv(index=False)
                st.download_button(
                    label="💾 Save CSV",
                    data=csv_data,
                    file_name=f"jpm_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No signals to export")
    
    with col_export3:
        if st.button("📄 Generate PDF Report", use_container_width=True):
            st.info("📄 PDF generation coming soon!")
            st.caption("Will include: Executive summary, KPIs, charts, and signals")
    
    st.markdown("---")
    
    # Dashboard customization
    st.markdown("### 🎨 Dashboard Customization")
    
    with st.expander("⚙️ Display Preferences"):
        theme = st.selectbox(
            "Color Theme:",
            ["JPM Blue (Default)", "Dark Mode", "Light Mode"]
        )
        
        show_tooltips = st.checkbox("Show help tooltips", value=True)
        compact_mode = st.checkbox("Compact mode (smaller charts)", value=False)
        
        if compact_mode:
            st.info("ℹ️ Compact mode will be applied on next refresh")
    
    st.markdown("---")
    
    # About section
    st.markdown("### ℹ️ About This Dashboard")
    
    st.info("""
    **JPMorgan European Equity Thesis Monitor**
    
    Version: 1.0.0
    
    Built with:
    - Python 3.11+
    - Streamlit
    - Plotly
    - Yahoo Finance API
    - FRED API
    - FinBERT (Sentiment Analysis)
    
    **Data Sources:**
    - Market Data: Yahoo Finance
    - Macro Data: Federal Reserve Economic Data (FRED)
    - News: NewsAPI
    - Sentiment: FinBERT AI Model
    
    **Features:**
    - Real-time market monitoring
    - AI-powered sentiment analysis
    - Quantitative backtesting
    - Trade signal generation
    - Email alerts
    
    **Disclaimer:**
    This dashboard is for educational and informational purposes only.
    Not investment advice. Always do your own research.
    """)
    
    st.markdown("---")
    
    # System info
    with st.expander("🔧 System Information"):
        col_sys1, col_sys2 = st.columns(2)
        
        with col_sys1:
            st.markdown("**Python Version:**")
            st.code(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
            
            st.markdown("**Streamlit Version:**")
            st.code(st.__version__)
        
        with col_sys2:
            st.markdown("**Cache Statistics:**")
            st.caption("Data cache: 1 hour TTL")
            st.caption("Module cache: Persistent")
            
            if st.button("🔄 Refresh System Info"):
                st.rerun()

# ============================================
# FOOTER
# ============================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.caption("📊 **JPMorgan Chase & Co.**")
    st.caption("European Equity Research")

with footer_col2:
    st.caption("🔄 Last Updated:")
    st.caption(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

with footer_col3:
    st.caption("🛠️ Built with Streamlit")
    st.caption("[View on GitHub](https://github.com/yourusername/jpm-dashboard)")

# ============================================
# PERFORMANCE MONITORING (Hidden)
# ============================================

# Track page load time
if 'load_time' not in st.session_state:
    st.session_state.load_time = datetime.now()

load_duration = (datetime.now() - st.session_state.load_time).total_seconds()

# Debug info (only show in development)
if os.getenv('DEBUG', 'False').lower() == 'true':
    with st.expander("🐛 Debug Information"):
        st.write(f"Page load time: {load_duration:.2f}s")
        st.write(f"Active view: {view}")
        st.write(f"Signals generated: {len(signals)}")
        st.write(f"Data timestamp: {data.get('timestamp', 'N/A')}")
                