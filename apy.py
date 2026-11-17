import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="NSE Breakout & Trend Screener", layout="wide", page_icon="📈")

st.title("📈 Institutional Breakout & Trend Screener (NSE)")
st.caption("Strategy: 125D High + Volume Surge + RSI < 70 + High Volatility + SMA 50/200 + EMA 21/50")

# --- Sidebar ---
st.sidebar.header("Stock Selection")
watchlist = [
    "TATAMOTORS", "RELIANCE", "TCS", "INFY", "HDFCBANK",
    "ICICIBANK", "SBIN", "BHARTIARTL", "ADANIENT", "TITAN",
    "ITC", "KOTAKBANK", "LT", "BAJFINANCE", "MARUTI"
]
selected_stock = st.sidebar.selectbox("Choose a Stock to Inspect:", watchlist)
custom_ticker = st.sidebar.text_input("Or Enter Custom Symbol (e.g. WIPRO, COALINDIA):").upper().strip()

stock_symbol = custom_ticker if custom_ticker else selected_stock
ticker = f"{stock_symbol}.NS" if not stock_symbol.endswith(".NS") else stock_symbol

@st.cache_data(ttl=300)
def fetch_and_calculate(sym):
    df = yf.download(sym, period="2y", interval="1d", progress=False)
    if df.empty or len(df) < 200:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.rename(columns=str.lower, inplace=True)

    # Rule A: 125D Breakout
    df['high_125'] = df['high'].shift(1).rolling(125).max()
    # Rule B: 125D Volume SMA
    df['vol_sma_125'] = df['volume'].rolling(125).mean()
    # Rule C: RSI 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    # Rule D: Volatility (ATR 14 vs 50D SMA ATR)
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
    return df.dropna()

data = fetch_and_calculate(ticker)

if data is None:
    st.error(f"Could not load sufficient historical data for **{ticker}**. Please verify the ticker.")
else:
    latest = data.iloc[-1]
    prev = data.iloc[-2]

    # Evaluate rules
    c_a = latest['close'] > latest['high_125']
    c_b = latest['volume'] > latest['vol_sma_125']
    c_c = latest['rsi'] < 70
    c_d = latest['atr'] > latest['atr_sma_50']
    c_hi = (latest['close'] > latest['sma_50']) and (latest['sma_50'] > latest['sma_200'])
    c_j = latest['ema_21'] > latest['ema_50']
    sell_crossover = (latest['ema_21'] < latest['ema_50']) and (prev['ema_21'] >= prev['ema_50'])

    score = sum([c_a, c_b, c_c, c_d, c_hi, c_j])
    close_price = round(latest['close'], 2)
    atr_val = round(latest['atr'], 2)

    if score == 6:
        signal = "STRONG BUY"
        signal_color = "#00C853"
        sl = round(close_price - (2 * atr_val), 2)
        tgt = round(close_price + (4 * atr_val), 2)
    elif sell_crossover:
        signal = "SELL / EXIT"
        signal_color = "#D50000"
        sl, tgt = "-", "-"
    else:
        signal = "HOLD / NEUTRAL"
        signal_color = "#FFAB00"
        sl, tgt = "-", "-"

    # Header Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"₹{close_price:,.2f}")
    m2.markdown(f"**Signal**  \n<h2 style='color:{signal_color};margin:0;'>{signal}</h2>", unsafe_allow_html=True)
    m3.metric("Calculated Stop-Loss", f"₹{sl}" if sl != "-" else "-")
    m4.metric("Calculated Target (1:2 RR)", f"₹{tgt}" if tgt != "-" else "-")

    st.divider()

    # Checklist
    st.subheader(f"Strategy Checklist: {score}/6 Conditions Met")
    r1, r2, r3 = st.columns(3)
    r4, r5, r6 = st.columns(3)
    r1.success("✅ A: 125D High Breakout" if c_a else f"❌ A: Below 125D High (₹{latest['high_125']:.2f})")
    r2.success("✅ B: High Volume Surge" if c_b else "❌ B: Below 125D Avg Volume")
    r3.success(f"✅ C: Safe RSI ({latest['rsi']:.1f} < 70)" if c_c else f"❌ C: Overbought RSI ({latest['rsi']:.1f})")
    r4.success("✅ D: Expanding Volatility" if c_d else "❌ D: Normal/Low Volatility")
    r5.success("✅ E-I: Price > 50 SMA > 200 SMA" if c_hi else "❌ E-I: Macro Trend Incomplete")
    r6.success("✅ J-K: 21 EMA > 50 EMA" if c_j else "❌ J-K: 21 EMA < 50 EMA")

    st.divider()

    # Charts
    st.subheader(f"Candlestick & Technical Analysis: {stock_symbol}")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])

    fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['high_125'], line=dict(color='orange', dash='dash'), name="125D Breakout"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['sma_50'], line=dict(color='yellow', width=1.5), name="50 SMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['sma_200'], line=dict(color='green', width=1.5), name="200 SMA"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['ema_21'], line=dict(color='blue', width=1.5), name="21 EMA (Blue)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['ema_50'], line=dict(color='red', width=1.5), name="50 EMA (Red)"), row=1, col=1)

    vol_colors = ['green' if c >= o else 'red' for c, o in zip(data['close'], data['open'])]
    fig.add_trace(go.Bar(x=data.index, y=data['volume'], marker_color=vol_colors, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['vol_sma_125'], line=dict(color='purple', width=1.5), name="125D Avg Vol"), row=2, col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data['rsi'], line=dict(color='magenta', width=1.5), name="RSI (14)"), row=3, col=1)
    fig.add_hline(y=70, line=dict(color='red', dash='dot'), row=3, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dot'), row=3, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
