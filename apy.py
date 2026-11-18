import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime
import pytz

st.set_page_config(page_title="NSE Multi-Timeframe Screener", layout="wide", page_icon="⚡")

IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

st.title("⚡ Multi-Timeframe Institutional Breakout Screener")

# ========================================================
# 1. TIMEFRAME TOGGLE SELECTOR
# ========================================================
selected_tf = st.radio(
    "⏱️ **Select Strategy Timeframe:**",
    (
        "📅 Daily (1D) — Positional Swing (125-Day High / Hold 2-4 Weeks)",
        "⏰ 1-Hour (1H) — Short-Term Swing (125-Hour High / Hold 2-5 Days)"
    ),
    horizontal=True
)

is_daily = "Daily" in selected_tf
interval = "1d" if is_daily else "1h"
period = "2y" if is_daily else "730d"
bar_label = "125-Day" if is_daily else "125-Hour"
time_desc = "6 Months" if is_daily else "20 Trading Days (~1 Month)"

st.caption(f"Strategy: {bar_label} Breakout ({time_desc}) + Volume Surge + RSI < 70 + Volatility + SMA 50/200 + EMA 21/50 | Updated: {now_ist.strftime('%d-%b-%Y %I:%M %p IST')}")

# Top 80 Liquid Stocks on NSE
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
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
    "ABB", "AMBUJACEM", "BANKBARODA", "BOSCHLTD", "CANBK",
    "CHOLAFIN", "DABUR", "DLF", "GAIL", "GODREJCP",
    "HAVELLS", "HAL", "ICICIGI", "IOC", "IRCTC", "IRFC",
    "JINDALSTEL", "JIOFIN", "MOTHERSON", "PIDILITIND", "PNB",
    "PFC", "RECLTD", "SIEMENS", "TORNTPHARM", "TVSMOTOR",
    "VEDL", "ZOMATO", "ZYDUSLIFE", "POLYCAB"
]

def analyze_stock(symbol, tf_interval, tf_period):
    try:
        ticker = f"{symbol}.NS"
        df = yf.download(ticker, period=tf_period, interval=tf_interval, progress=False)
        if df.empty or len(df) < 200:
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

        # SMA 50 & 200
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()

        # EMA 21 & 50
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        df.dropna(inplace=True)
        if len(df) < 2:
            return None

        row = df.iloc[-1]
        prev = df.iloc[-2]

        # Stage 1: Core Breakout & Volume Trigger
        cond_a_breakout = row['close'] > row['high_125']
        cond_b_volume = row['volume'] > row['vol_sma_125']
        primary_breakout = cond_a_breakout and cond_b_volume

        # Stage 2: Quality Confirmations
        cond_c_rsi = row['rsi'] < 70
        cond_d_volatility = row['atr'] > row['atr_sma_50']
        cond_e_sma = (row['close'] > row['sma_50']) and (row['sma_50'] > row['sma_200'])
        cond_f_ema = row['ema_21'] > row['ema_50']

        # Sell Signal
        is_sell = (row['ema_21'] < row['ema_50']) and (prev['ema_21'] >= prev['ema_50'])

        vol_ratio = round(row['volume'] / (row['vol_sma_125'] + 1e-9), 2)
        close = round(row['close'], 2)
        atr_val = round(row['atr'], 2)

        confirm_score = sum([cond_c_rsi, cond_d_volatility, cond_e_sma, cond_f_ema])

        if primary_breakout:
            quality = "⭐⭐⭐ HIGH CONVICTION" if confirm_score == 4 else ("⭐⭐ GOOD SETUP" if confirm_score >= 2 else "⭐ SPECULATIVE")
            return {
                "Type": "BREAKOUT",
                "Stock": symbol,
                "Price": f"₹{close:,.2f}",
                f"{bar_label} High": f"₹{round(row['high_125'], 2):,.2f}",
                "Volume Surge": f"{vol_ratio}x Avg",
                "Quality Rating": quality,
                "Stop Loss": f"₹{round(close - (2 * atr_val), 2):,.2f}",
                "Target (1:2 RR)": f"₹{round(close + (4 * atr_val), 2):,.2f}",
                "RSI (<70)": f"PASS ({row['rsi']:.1f})" if cond_c_rsi else f"FAIL ({row['rsi']:.1f})",
                "50 > 200 SMA": "PASS" if cond_e_sma else "FAIL",
                "21 > 50 EMA": "PASS" if cond_f_ema else "FAIL"
            }
        elif is_sell:
            return {
                "Type": "SELL",
                "Stock": symbol,
                "Price": f"₹{close:,.2f}",
                "Reason": "21 EMA crossed below 50 EMA",
                "RSI": round(row['rsi'], 1)
            }
        return None
    except:
        return None

@st.cache_data(ttl=900)
def run_scan(tf_interval, tf_period):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_stock, sym, tf_interval, tf_period): sym for sym in NIFTY_UNIVERSE}
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    breakouts = [r for r in results if r and r["Type"] == "BREAKOUT"]
    sells = [r for r in results if r and r["Type"] == "SELL"]
    return breakouts, sells

# Top Actions
c1, c2 = st.columns([3, 1])
with c2:
    if st.button("🔄 Force Re-Scan Now", type="primary"):
        st.cache_data.clear()
        st.rerun()

with st.spinner(f"Scanning market on {selected_tf.split('—')[0].strip()} timeframe..."):
    breakouts, sells = run_scan(interval, period)

# ========================================================
# SECTION 1: PRIMARY BREAKOUT ALERTS
# ========================================================
st.subheader(f"🎯 Primary Breakout Alerts ({bar_label} High + Volume Surge)")
st.write(f"Showing stocks that broke above their previous {bar_label} high with above-average volume on the **{selected_tf.split('—')[0].strip()}** chart.")

if breakouts:
    df_breakouts = pd.DataFrame(breakouts).drop(columns=["Type"])
    st.dataframe(df_breakouts, use_container_width=True)
else:
    st.info(f"ℹ️ No stocks currently meeting {bar_label} breakout + volume surge on this timeframe.")

st.divider()

# ========================================================
# SECTION 2: SELL / EXIT SIGNALS
# ========================================================
st.subheader(f"🔴 Trend Exit Alerts (21 EMA < 50 EMA on {selected_tf.split('—')[0].strip()})")
if sells:
    df_sells = pd.DataFrame(sells).drop(columns=["Type"])
    st.dataframe(df_sells, use_container_width=True)
else:
    st.info(f"ℹ️ No fresh exit signals on the {selected_tf.split('—')[0].strip()} timeframe.")

st.divider()

# ========================================================
# SECTION 3: INTERACTIVE CHART VIEWER
# ========================================================
with st.expander(f"🔍 Open {selected_tf.split('—')[0].strip()} Chart for Any Stock"):
    selected = st.selectbox("Choose stock to review chart:", NIFTY_UNIVERSE)
    if selected:
        df_c = yf.download(f"{selected}.NS", period=period, interval=interval, progress=False)
        if not df_c.empty:
            if isinstance(df_c.columns, pd.MultiIndex):
                df_c.columns = df_c.columns.get_level_values(0)
            df_c.rename(columns=str.lower, inplace=True)
            df_c['high_125'] = df_c['high'].shift(1).rolling(125).max()
            df_c['vol_125'] = df_c['volume'].rolling(125).mean()
            df_c['ema_21'] = df_c['close'].ewm(span=21, adjust=False).mean()
            df_c['ema_50'] = df_c['close'].ewm(span=50, adjust=False).mean()
            df_c['sma_50'] = df_c['close'].rolling(50).mean()
            df_c['sma_200'] = df_c['close'].rolling(200).mean()

            # Candlestick Subplots
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['open'], high=df_c['high'], low=df_c['low'], close=df_c['close'], name=f"Price ({interval})"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['high_125'], line=dict(color='orange', dash='dash'), name=f"{bar_label} High"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['ema_21'], line=dict(color='blue', width=1.5), name="21 EMA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['ema_50'], line=dict(color='red', width=1.5), name="50 EMA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['sma_50'], line=dict(color='yellow', width=1.5), name="50 SMA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['sma_200'], line=dict(color='green', width=1.5), name="200 SMA"), row=1, col=1)

            colors = ['green' if c >= o else 'red' for c, o in zip(df_c['close'], df_c['open'])]
            fig.add_trace(go.Bar(x=df_c.index, y=df_c['volume'], marker_color=colors, name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['vol_125'], line=dict(color='purple'), name="125-Bar Avg Vol"), row=2, col=1)

            fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
