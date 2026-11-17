import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime

st.set_page_config(page_title="NSE Live Signals (NIFTY 100)", layout="wide", page_icon="⚡")

st.title("⚡ NSE Actionable Signals Dashboard")
st.caption(f"Real-time Strategy: 125D High Breakout + Volume Surge (> 125D Avg) + RSI < 70 + Volatility + SMA 50/200 + EMA 21/50 | Updated: {datetime.now().strftime('%d-%b-%Y %H:%M')}")

# --- Official Liquid NIFTY 100 Stock List ---
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
    "HAVELLS", "HAL", "ICICIGI", "ICICIPRULI", "IOC",
    "IRCTC", "IRFC", "JINDALSTEL", "JIOFIN", "LICI",
    "MOTHERSON", "PIDILITIND", "PNB", "PFC", "RECLTD",
    "SIEMENS", "TORNTPHARM", "TVSMOTOR", "UNITDSPR", "VBL",
    "VEDL", "ZOMATO", "ZYDUSLIFE", "POLYCAB", "PERSISTENT"
]

# --- Strategy Calculator ---
def process_stock(symbol):
    try:
        ticker = f"{symbol}.NS"
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty or len(df) < 200:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.rename(columns=str.lower, inplace=True)

        # Rule A: 125D High Breakout
        df['high_125'] = df['high'].shift(1).rolling(125).max()

        # Rule B: 125D Average Daily Volume
        df['vol_sma_125'] = df['volume'].rolling(125).mean()

        # Rule C: RSI 14 (< 70)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Rule D: Volatility (ATR 14 vs 50D ATR SMA)
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_sma_50'] = df['atr'].rolling(50).mean()

        # Rules E-I: Macro Trend SMA 50 & SMA 200
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()

        # Rules J-K: Momentum EMA 21 & EMA 50
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        df.dropna(inplace=True)
        if len(df) < 2:
            return None

        row = df.iloc[-1]
        prev = df.iloc[-2]

        # Checklist Rules
        c_a = row['close'] > row['high_125']
        c_b = row['volume'] > row['vol_sma_125']
        c_c = row['rsi'] < 70
        c_d = row['atr'] > row['atr_sma_50']
        c_hi = (row['close'] > row['sma_50']) and (row['sma_50'] > row['sma_200'])
        c_j = row['ema_21'] > row['ema_50']

        # Sell Rule: 21 EMA crosses below 50 EMA
        is_sell = (row['ema_21'] < row['ema_50']) and (prev['ema_21'] >= prev['ema_50'])

        # Volume Multiplier (e.g. 1.8x average volume)
        vol_surge_ratio = round(row['volume'] / (row['vol_sma_125'] + 1e-9), 2)
        score = sum([c_a, c_b, c_c, c_d, c_hi, c_j])

        close = round(row['close'], 2)
        atr_val = round(row['atr'], 2)

        if score == 6:
            return {
                "Type": "BUY",
                "Stock": symbol,
                "Current Price": f"₹{close:,.2f}",
                "Stop Loss": f"₹{round(close - (2 * atr_val), 2):,.2f}",
                "Target (1:2 RR)": f"₹{round(close + (4 * atr_val), 2):,.2f}",
                "Volume Surge": f"{vol_surge_ratio}x Avg",
                "RSI": round(row['rsi'], 1),
                "Breakout Level": f"₹{round(row['high_125'], 2):,.2f}",
                "Score": f"{score}/6"
            }
        elif is_sell:
            return {
                "Type": "SELL",
                "Stock": symbol,
                "Current Price": f"₹{close:,.2f}",
                "Reason": "21 EMA crossed below 50 EMA",
                "Volume Today": f"{int(row['volume']):,}",
                "RSI": round(row['rsi'], 1)
            }
        return None
    except:
        return None

# --- Cached Auto-Scanner Engine (Runs on page load / refresh) ---
@st.cache_data(ttl=900)  # Automatically re-scans fresh market data every 15 minutes
def run_full_scan():
    buy_signals = []
    sell_signals = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_stock, NIFTY_UNIVERSE))

    for item in results:
        if item is not None:
            if item["Type"] == "BUY":
                buy_signals.append(item)
            elif item["Type"] == "SELL":
                sell_signals.append(item)

    return buy_signals, sell_signals

# Quick Refresh Button
top_c1, top_c2 = st.columns([3, 1])
with top_c2:
    if st.button("🔄 Force Re-Scan Now", type="primary"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Analyzing daily trends and volume surges across the market..."):
    buys, sells = run_full_scan()

# --- 1. ACTIVE BUY SIGNALS ---
st.subheader("🟢 Active BUY Signals (All 6 Rules + Volume Surge Matched)")
if buys:
    buys_df = pd.DataFrame(buys).drop(columns=["Type"])
    st.dataframe(buys_df, use_container_width=True)
else:
    st.info("ℹ️ No stocks meeting ALL 6 breakout conditions right now. (Breakouts with volume expansion are selective to protect your capital).")

st.divider()

# --- 2. ACTIVE SELL SIGNALS ---
st.subheader("🔴 Active SELL / EXIT Signals (Trend Breakdown)")
if sells:
    sells_df = pd.DataFrame(sells).drop(columns=["Type"])
    st.dataframe(sells_df, use_container_width=True)
else:
    st.info("ℹ️ No fresh sell crossover signals detected today.")

st.divider()

# --- 3. ON-DEMAND CHART VIEWER ---
with st.expander("📊 View Detailed Chart for Any Specific Stock"):
    inspect_stock = st.selectbox("Select Stock to Inspect Chart:", NIFTY_UNIVERSE)
    if inspect_stock:
        df_chart = yf.download(f"{inspect_stock}.NS", period="1y", interval="1d", progress=False)
        if not df_chart.empty:
            if isinstance(df_chart.columns, pd.MultiIndex):
                df_chart.columns = df_chart.columns.get_level_values(0)
            df_chart.rename(columns=str.lower, inplace=True)
            df_chart['high_125'] = df_chart['high'].shift(1).rolling(125).max()
            df_chart['vol_125'] = df_chart['volume'].rolling(125).mean()
            df_chart['ema_21'] = df_chart['close'].ewm(span=21, adjust=False).mean()
            df_chart['ema_50'] = df_chart['close'].ewm(span=50, adjust=False).mean()

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['open'], high=df_chart['high'], low=df_chart['low'], close=df_chart['close'], name="Candles"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['high_125'], line=dict(color='orange', dash='dash'), name="125D High"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['ema_21'], line=dict(color='blue'), name="21 EMA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['ema_50'], line=dict(color='red'), name="50 EMA"), row=1, col=1)

            colors = ['green' if c >= o else 'red' for c, o in zip(df_chart['close'], df_chart['open'])]
            fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['volume'], marker_color=colors, name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['vol_125'], line=dict(color='purple'), name="125D Avg Vol"), row=2, col=1)

            fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
