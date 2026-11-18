import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime
import pytz

st.set_page_config(page_title="NSE Volume & Breakout Tracker", layout="wide", page_icon="⚡")

IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

st.title("⚡ NSE Institutional Volume & Breakout Tracker")

# ========================================================
# 1. TIMEFRAME TOGGLE
# ========================================================
selected_tf = st.radio(
    "⏱️ **Select Strategy Timeframe:**",
    (
        "📅 Daily (1D) — Positional Swing (125-Day High / 6 Months)",
        "⏰ 1-Hour (1H) — Short-Term Swing (125-Hour High / ~20 Days)"
    ),
    horizontal=True
)

is_daily = "Daily" in selected_tf
interval = "1d" if is_daily else "1h"
# 60 days on 1h has ~375 hourly candles: plenty for 125 high + 200 SMA, and 10x faster
period = "2y" if is_daily else "60d" 
bar_label = "125-Day" if is_daily else "125-Hour"

st.caption(f"Strategy: {bar_label} High + Volume Surge + RSI < 70 + Volatility + SMA 50/200 + EMA 21/50 | Updated: {now_ist.strftime('%d-%b-%Y %I:%M %p IST')}")

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

        df.dropna(subset=['high_125', 'vol_sma_125', 'rsi', 'atr'], inplace=True)
        if len(df) < 2:
            return None

        row = df.iloc[-1]
        prev = df.iloc[-2]

        close = round(row['close'], 2)
        high_125 = round(row['high_125'], 2)
        atr_val = round(row['atr'], 2)
        vol_surge_ratio = round(row['volume'] / (row['vol_sma_125'] + 1e-9), 2)
        pct_from_high = round(((close - high_125) / high_125) * 100, 2)

        # Rules
        cond_a_breakout = close >= high_125
        cond_b_volume = vol_surge_ratio >= 1.0
        cond_c_rsi = row['rsi'] < 70
        cond_d_volatility = row['atr'] > row['atr_sma_50']
        cond_e_sma = row['close'] > row['sma_50']
        cond_f_ema = row['ema_21'] > row['ema_50']
        
        is_sell = (row['ema_21'] < row['ema_50']) and (prev['ema_21'] >= prev['ema_50'])
        confirm_score = sum([cond_c_rsi, cond_d_volatility, cond_e_sma, cond_f_ema])

        return {
            "Stock": symbol,
            "Price": f"₹{close:,.2f}",
            "Raw_Price": close,
            f"{bar_label} High": f"₹{high_125:,.2f}",
            "Distance to High": f"{pct_from_high:+0.2f}%",
            "Raw_Distance": pct_from_high,
            "Volume Surge": f"{vol_surge_ratio}x Avg",
            "Raw_Volume": vol_surge_ratio,
            "RSI": round(row['rsi'], 1),
            "Score": f"{confirm_score + (1 if cond_a_breakout else 0) + (1 if cond_b_volume else 0)}/6",
            "Is_Breakout": cond_a_breakout and cond_b_volume,
            "Is_Near_Breakout": (-3.0 <= pct_from_high < 0.0) and (vol_surge_ratio >= 0.8),
            "Is_High_Volume": vol_surge_ratio >= 1.2,
            "Is_Sell": is_sell,
            "Stop_Loss": f"₹{round(close - (2 * atr_val), 2):,.2f}",
            "Target": f"₹{round(close + (4 * atr_val), 2):,.2f}"
        }
    except:
        return None

@st.cache_data(ttl=600)
def run_scan(tf_interval, tf_period):
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(analyze_stock, sym, tf_interval, tf_period): sym for sym in NIFTY_UNIVERSE}
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return [r for r in results if r is not None]

# Action Button
c1, c2 = st.columns([3, 1])
with c2:
    if st.button("🔄 Force Re-Scan Now", type="primary"):
        st.cache_data.clear()
        st.rerun()

with st.spinner(f"Analyzing all 80 stocks on {selected_tf.split('—')[0].strip()}..."):
    all_results = run_scan(interval, period)

# ========================================================
# SECTION 1: CONFIRMED BREAKOUTS (Rules A + B)
# ========================================================
st.subheader(f"🎯 1. Active Breakouts ({bar_label} High + Volume Surge)")
breakouts = [r for r in all_results if r["Is_Breakout"]]
if breakouts:
    df_bo = pd.DataFrame(breakouts)[[
        "Stock", "Price", f"{bar_label} High", "Volume Surge", "Score", "Stop_Loss", "Target", "RSI"
    ]]
    st.dataframe(df_bo, use_container_width=True)
else:
    st.info(f"ℹ️ No stocks currently at exact 0% breakout with volume surge on the {selected_tf.split('—')[0].strip()} chart.")

st.divider()

# ========================================================
# SECTION 2: VOLUME SURGE MOVERS (Top Institutional Accumulation)
# ========================================================
st.subheader("⚡ 2. Top Volume Movers Today (Institutional Buying Radar)")
st.write("Stocks where current volume is substantially higher than the 125-bar average, showing active money inflow:")
vol_movers = sorted([r for r in all_results if r["Is_High_Volume"]], key=lambda x: x["Raw_Volume"], reverse=True)
if vol_movers:
    df_vol = pd.DataFrame(vol_movers)[[
        "Stock", "Price", "Volume Surge", "Distance to High", "RSI", "Score"
    ]]
    st.dataframe(df_vol, use_container_width=True)
else:
    st.info("ℹ️ No unusual volume surges (>1.2x average) detected at this moment.")

st.divider()

# ========================================================
# SECTION 3: NEAR BREAKOUT RADAR (Approaching the High)
# ========================================================
st.subheader(f"🔭 3. Near-Breakout Radar (Within 3% of {bar_label} High)")
st.write("Stocks setting up right below the breakout ceiling with strong volume — get ready before they cross:")
near_bo = sorted([r for r in all_results if r["Is_Near_Breakout"]], key=lambda x: x["Raw_Distance"], reverse=True)
if near_bo:
    df_near = pd.DataFrame(near_bo)[[
        "Stock", "Price", f"{bar_label} High", "Distance to High", "Volume Surge", "RSI", "Score"
    ]]
    st.dataframe(df_near, use_container_width=True)
else:
    st.info(f"ℹ️ No stocks currently within 3% of their {bar_label} high.")

st.divider()

# ========================================================
# SECTION 4: SELL / TREND EXIT ALERTS
# ========================================================
st.subheader("🔴 4. Trend Exit Alerts (21 EMA < 50 EMA)")
sells = [r for r in all_results if r["Is_Sell"]]
if sells:
    df_sells = pd.DataFrame(sells)[["Stock", "Price", "RSI"]]
    st.dataframe(df_sells, use_container_width=True)
else:
    st.info("ℹ️ No fresh sell crossovers on this timeframe.")

st.divider()

# ========================================================
# SECTION 5: INTERACTIVE CHART VIEWER
# ========================================================
with st.expander("📊 Open Chart to Inspect Price & Volume Bars"):
    inspect_sym = st.selectbox("Choose stock to view chart:", NIFTY_UNIVERSE)
    if inspect_sym:
        df_c = yf.download(f"{inspect_sym}.NS", period=period, interval=interval, progress=False)
        if not df_c.empty:
            if isinstance(df_c.columns, pd.MultiIndex):
                df_c.columns = df_c.columns.get_level_values(0)
            df_c.rename(columns=str.lower, inplace=True)
            df_c['high_125'] = df_c['high'].shift(1).rolling(125).max()
            df_c['vol_125'] = df_c['volume'].rolling(125).mean()
            df_c['ema_21'] = df_c['close'].ewm(span=21, adjust=False).mean()
            df_c['ema_50'] = df_c['close'].ewm(span=50, adjust=False).mean()

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['open'], high=df_c['high'], low=df_c['low'], close=df_c['close'], name="Candles"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['high_125'], line=dict(color='orange', dash='dash'), name=f"{bar_label} High"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['ema_21'], line=dict(color='blue'), name="21 EMA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['ema_50'], line=dict(color='red'), name="50 EMA"), row=1, col=1)

            colors = ['green' if c >= o else 'red' for c, o in zip(df_c['close'], df_c['open'])]
            fig.add_trace(go.Bar(x=df_c.index, y=df_c['volume'], marker_color=colors, name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['vol_125'], line=dict(color='purple'), name="125-Avg Vol"), row=2, col=1)

            fig.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
