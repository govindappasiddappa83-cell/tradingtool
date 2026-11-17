import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime
import pytz

st.set_page_config(page_title="NSE Next-Day Trade Planner", layout="wide", page_icon="📋")

IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

st.title("📋 NSE Next-Day Actionable Trade Planner")
st.caption(f"Strategy: 125D High Breakout + Volume Surge + RSI < 70 + High Volatility + SMA 50/200 + EMA 21/50 | Updated: {now_ist.strftime('%d-%b-%Y %I:%M %p IST')}")

# Top 80 Liquid NSE Stocks for Swing Trading
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

def analyze_stock(symbol):
    try:
        ticker = f"{symbol}.NS"
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty or len(df) < 200:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.rename(columns=str.lower, inplace=True)

        # Rule A: 125D High of previous sessions
        df['high_125'] = df['high'].shift(1).rolling(125).max()

        # Rule B: 125D Average Daily Volume
        df['vol_sma_125'] = df['volume'].rolling(125).mean()

        # Rule C: RSI 14 (Must be below 70)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))

        # Rule D: Expanding Volatility (ATR 14 vs 50D ATR SMA)
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_sma_50'] = df['atr'].rolling(50).mean()

        # Rules E-I: 50 & 200 SMA
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()

        # Rules J-K: 21 & 50 EMA
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        df.dropna(inplace=True)
        if len(df) < 2:
            return None

        row = df.iloc[-1]
        prev = df.iloc[-2]

        c_a = row['close'] > row['high_125']
        c_b = row['volume'] > row['vol_sma_125']
        c_c = row['rsi'] < 70
        c_d = row['atr'] > row['atr_sma_50']
        c_hi = (row['close'] > row['sma_50']) and (row['sma_50'] > row['sma_200'])
        c_j = row['ema_21'] > row['ema_50']
        
        is_sell = (row['ema_21'] < row['ema_50']) and (prev['ema_21'] >= prev['ema_50'])

        vol_surge = round(row['volume'] / (row['vol_sma_125'] + 1e-9), 2)
        score = sum([c_a, c_b, c_c, c_d, c_hi, c_j])

        close = round(row['close'], 2)
        atr_val = round(row['atr'], 2)

        if score == 6:
            return {
                "Type": "BUY",
                "Stock": symbol,
                "Close Price": f"₹{close:,.2f}",
                "Breakout Level": f"₹{round(row['high_125'], 2):,.2f}",
                "Stop Loss (SL)": f"₹{round(close - (2 * atr_val), 2):,.2f}",
                "Target (1:2 RR)": f"₹{round(close + (4 * atr_val), 2):,.2f}",
                "Volume Surge": f"{vol_surge}x Avg",
                "RSI": round(row['rsi'], 1)
            }
        elif is_sell:
            return {
                "Type": "SELL",
                "Stock": symbol,
                "Close Price": f"₹{close:,.2f}",
                "Reason": "21 EMA crossed below 50 EMA",
                "RSI": round(row['rsi'], 1)
            }
        return None
    except:
        return None

@st.cache_data(ttl=1800) # Cache for 30 mins
def run_scan():
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(analyze_stock, NIFTY_UNIVERSE))
    buys = [r for r in results if r and r["Type"] == "BUY"]
    sells = [r for r in results if r and r["Type"] == "SELL"]
    return buys, sells

# Top action bar
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Force Re-Scan Now", type="primary"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Analyzing closed daily candles across NSE..."):
    buys, sells = run_scan()

# --- SECTION 1: CONFIRMED BUY SIGNALS FOR TOMORROW ---
st.subheader("🟢 Confirmed Breakout Setups (Buy Consideration for Next Session)")
st.write("These stocks closed today above their 125-day high with institutional volume surge and bullish trend alignment.")

if buys:
    df_buys = pd.DataFrame(buys).drop(columns=["Type"])
    st.dataframe(df_buys, use_container_width=True)
    st.info("💡 **How to trade manually tomorrow:** Open your Zerodha Kite app before 9:15 AM. You can place an AMO (After Market Order) or enter around market open if price stays above the Breakout Level. Keep the Stop Loss strictly active.")
else:
    st.info("ℹ️ No stocks passed all 6 breakout conditions on today's close. No new buy setups recommended for tomorrow.")

st.divider()

# --- SECTION 2: SELL / EXIT SIGNALS ---
st.subheader("🔴 Trend Breakdown (Exit Signals)")
st.write("These stocks saw their 21 EMA cross below their 50 EMA today. If you hold open positions, consider closing or tightening stops.")

if sells:
    df_sells = pd.DataFrame(sells).drop(columns=["Type"])
    st.dataframe(df_sells, use_container_width=True)
else:
    st.info("ℹ️ No fresh exit signals detected today.")

st.divider()

# --- SECTION 3: INTERACTIVE CHART CHECKER ---
with st.expander("🔍 Inspect Chart for Any Stock Before Placing Trade"):
    selected = st.selectbox("Choose stock to review chart:", NIFTY_UNIVERSE)
    if selected:
        df_c = yf.download(f"{selected}.NS", period="1y", interval="1d", progress=False)
        if not df_c.empty:
            if isinstance(df_c.columns, pd.MultiIndex):
                df_c.columns = df_c.columns.get_level_values(0)
            df_c.rename(columns=str.lower, inplace=True)
            df_c['high_125'] = df_c['high'].shift(1).rolling(125).max()
            df_c['vol_125'] = df_c['volume'].rolling(125).mean()
            df_c['ema_21'] = df_c['close'].ewm(span=21, adjust=False).mean()
            df_c['ema_50'] = df_c['close'].ewm(span=50, adjust=False).mean()

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            fig.add_trace(go.Candlestick(x=df_c.index, open=df_c['open'], high=df_c['high'], low=df_c['low'], close=df_c['close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['high_125'], line=dict(color='orange', dash='dash'), name="125D High Breakout"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['ema_21'], line=dict(color='blue'), name="21 EMA (Blue)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['ema_50'], line=dict(color='red'), name="50 EMA (Red)"), row=1, col=1)

            colors = ['green' if c >= o else 'red' for c, o in zip(df_c['close'], df_c['open'])]
            fig.add_trace(go.Bar(x=df_c.index, y=df_c['volume'], marker_color=colors, name="Volume"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df_c.index, y=df_c['vol_125'], line=dict(color='purple'), name="125D Avg Vol"), row=2, col=1)

            fig.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
