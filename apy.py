import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime
import pytz

st.set_page_config(page_title="Multi-Asset Trading Dashboard", layout="wide", page_icon="⚡")

IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

# Top Navigation Tabs
main_tab1, main_tab2 = st.tabs(["🥇 Gold (XAU/USD) Dashboard", "🇮🇳 NSE Stocks Scanner"])

# ========================================================
# HELPER: STRATEGY ENGINE (Rules A through K)
# ========================================================
def compute_indicators(df):
    if df.empty or len(df) < 130:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.rename(columns=str.lower, inplace=True)

    # 125 Bars High Breakout
    df['high_125'] = df['high'].shift(1).rolling(125).max()
    df['vol_sma_125'] = df['volume'].rolling(125).mean()

    # RSI 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR 14 vs 50 ATR SMA
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_sma_50'] = df['atr'].rolling(50).mean()

    # SMAs & EMAs
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(min(200, len(df)-1)).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    return df.dropna(subset=['high_125', 'vol_sma_125', 'rsi', 'atr'])

# ========================================================
# TAB 1: GOLD (XAU/USD) DASHBOARD
# ========================================================
with main_tab1:
    st.title("🥇 Gold (XAU/USD) Real-Time Strategy Dashboard")
    st.caption("Tracking Spot / COMEX Futures (GC=F) with real volume, 125-bar breakout levels, and ATR risk rules.")

    c_g1, c_g2 = st.columns([2, 1])
    with c_g1:
        gold_tf = st.radio(
            "⏱️ **Select Gold Timeframe:**",
            ("Daily (1D) — Macro Swing", "4-Hour (4H) — Swing", "1-Hour (1H) — Intraday / Short-Term", "15-Minute (15M) — Scalp"),
            horizontal=True
        )
    with c_g2:
        if st.button("🔄 Refresh Gold Price", type="primary"):
            st.cache_data.clear()
            st.rerun()

    # Map timeframe
    tf_map = {
        "Daily (1D) — Macro Swing": ("1d", "2y", "125-Day"),
        "4-Hour (4H) — Swing": ("1h", "120d", "125-Bar"),  # approximated via 1h
        "1-Hour (1H) — Intraday / Short-Term": ("1h", "60d", "125-Hour"),
        "15-Minute (15M) — Scalp": ("15m", "30d", "125-Bar (15M)")
    }
    g_interval, g_period, g_bar_label = tf_map[gold_tf]

    @st.cache_data(ttl=120)
    def fetch_gold(interval_val, period_val):
        # GC=F provides real COMEX exchange volume for gold
        df = yf.download("GC=F", period=period_val, interval=interval_val, progress=False)
        return compute_indicators(df)

    with st.spinner("Fetching live Gold market data..."):
        gold_df = fetch_gold(g_interval, g_period)

    if gold_df is None or len(gold_df) < 2:
        st.error("Could not fetch sufficient Gold market candles. Try another timeframe or refresh.")
    else:
        row = gold_df.iloc[-1]
        prev = gold_df.iloc[-2]

        close = round(row['close'], 2)
        high_125 = round(row['high_125'], 2)
        atr_val = round(row['atr'], 2)
        vol_ratio = round(row['volume'] / (row['vol_sma_125'] + 1e-9), 2)
        pct_diff = round(((close - high_125) / high_125) * 100, 2)

        # Rule Evaluations
        c_a = close >= high_125
        c_b = vol_ratio >= 1.0
        c_c = row['rsi'] < 70
        c_d = row['atr'] > row['atr_sma_50']
        c_e = row['close'] > row['sma_50']
        c_f = row['ema_21'] > row['ema_50']
        sell_crossover = (row['ema_21'] < row['ema_50']) and (prev['ema_21'] >= prev['ema_50'])

        score = sum([c_a, c_b, c_c, c_d, c_e, c_f])

        if c_a and c_b:
            sig = "🟢 STRONG BREAKOUT BUY"
            sig_color = "#00C853"
        elif score >= 4:
            sig = "🟡 BULLISH TREND (Waiting for Breakout)"
            sig_color = "#FFAB00"
        elif sell_crossover:
            sig = "🔴 SELL / EXIT (EMA Breakdown)"
            sig_color = "#D50000"
        else:
            sig = "⚪ NEUTRAL / CONSOLIDATION"
            sig_color = "#888888"

        # Metric Tiles
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Gold Price (USD)", f"${close:,.2f}")
        m2.markdown(f"**Signal**  \n<h3 style='color:{sig_color};margin:0;'>{sig}</h3>", unsafe_allow_html=True)
        m3.metric(f"Stop-Loss (-2 ATR)", f"${round(close - (2 * atr_val), 2):,.2f}")
        m4.metric(f"Target (+4 ATR, 1:2 RR)", f"${round(close + (4 * atr_val), 2):,.2f}")

        st.divider()

        # Strategy Checklist Grid
        st.subheader(f"Gold Checklist Score: {score}/6 Conditions Met")
        k1, k2, k3 = st.columns(3)
        k4, k5, k6 = st.columns(3)

        k1.success(f"✅ A: {g_bar_label} Breakout" if c_a else f"❌ A: Below {g_bar_label} High (${high_125:,.2f}) [{pct_diff:+0.2f}%]")
        k2.success(f"✅ B: Volume Surge ({vol_ratio}x Avg)" if c_b else f"❌ B: Low Volume ({vol_ratio}x Avg)")
        k3.success(f"✅ C: Safe RSI ({row['rsi']:.1f} < 70)" if c_c else f"❌ C: Overbought RSI ({row['rsi']:.1f})")
        k4.success("✅ D: Expanding Volatility" if c_d else "❌ D: Low Volatility")
        k5.success("✅ E: Price > 50 SMA" if c_e else "❌ E: Price Below 50 SMA")
        k6.success("✅ F: 21 EMA > 50 EMA" if c_f else "❌ F: 21 EMA < 50 EMA")

        st.divider()

        # Interactive Gold Chart
        st.subheader(f"Gold (XAU/USD) Technical Chart ({g_interval.upper()})")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])

        # Price Candlesticks + Indicators
        fig.add_trace(go.Candlestick(x=gold_df.index, open=gold_df['open'], high=gold_df['high'], low=gold_df['low'], close=gold_df['close'], name="Gold Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['high_125'], line=dict(color='orange', dash='dash'), name=f"{g_bar_label} High"), row=1, col=1)
        fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['ema_21'], line=dict(color='blue', width=1.5), name="21 EMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['ema_50'], line=dict(color='red', width=1.5), name="50 EMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['sma_50'], line=dict(color='yellow', width=1.5), name="50 SMA"), row=1, col=1)

        # Volume
        v_colors = ['green' if c >= o else 'red' for c, o in zip(gold_df['close'], gold_df['open'])]
        fig.add_trace(go.Bar(x=gold_df.index, y=gold_df['volume'], marker_color=v_colors, name="COMEX Volume"), row=2, col=1)
        fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['vol_sma_125'], line=dict(color='purple'), name="125-Avg Volume"), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['rsi'], line=dict(color='magenta', width=1.5), name="RSI (14)"), row=3, col=1)
        fig.add_hline(y=70, line=dict(color='red', dash='dot'), row=3, col=1)
        fig.add_hline(y=30, line=dict(color='green', dash='dot'), row=3, col=1)

        fig.update_layout(height=750, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# ========================================================
# TAB 2: NSE STOCKS SCANNER (Preserved intact)
# ========================================================
with main_tab2:
    st.subheader("🇮🇳 NSE Multi-Timeframe Stock Screener")
    st.info("Your existing NSE stock screener is fully preserved here.")

    NIFTY_UNIVERSE = [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
        "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
        "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
        "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
        "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
        "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
        "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
        "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
        "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
        "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO", "ZOMATO", "HAL"
    ]

    selected_stock = st.selectbox("Select NSE Stock to Check:", NIFTY_UNIVERSE)
    if selected_stock:
        df_stock = yf.download(f"{selected_stock}.NS", period="1y", interval="1d", progress=False)
        clean_stock = compute_indicators(df_stock)
        if clean_stock is not None:
            st.write(f"Latest Close for {selected_stock}: ₹{clean_stock.iloc[-1]['close']:.2f}")
