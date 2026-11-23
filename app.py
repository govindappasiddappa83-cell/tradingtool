import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime
import pytz

st.set_page_config(page_title="Professional Trading Dashboard", layout="wide", page_icon="⚡")
IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

main_tab1, main_tab2 = st.tabs(["🥇 Gold (XAU/USD) Dashboard", "🇮🇳 NSE Stocks Scanner"])

def compute_indicators(df):
    if df.empty or len(df) < 200: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.rename(columns=str.lower, inplace=True)
    
    # Existing Indicators
    df['high_125'] = df['high'].shift(1).rolling(125).max()
    df['low_125'] = df['low'].shift(1).rolling(125).min()
    df['vol_sma_125'] = df['volume'].rolling(125).mean()
    
    # New Volatility Filter (Bollinger Width)
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_width'] = (df['bb_mid'] + 2*df['bb_std']) - (df['bb_mid'] - 2*df['bb_std'])
    
    # New Trend Filter (200 SMA)
    df['sma_200'] = df['close'].rolling(200).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI & ATR
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_sma_50'] = df['atr'].rolling(50).mean()
    
    return df.dropna()

NIFTY_UNIVERSE = ["ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL", "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC", "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO", "ABB", "AMBUJACEM", "BANKBARODA", "BOSCHLTD", "CANBK", "CHOLAFIN", "DABUR", "DLF", "GAIL", "GODREJCP", "HAVELLS", "HAL", "ICICIGI", "IOC", "IRCTC", "IRFC", "JINDALSTEL", "JIOFIN", "MOTHERSON", "PIDILITIND", "PNB", "PFC", "RECLTD", "SIEMENS", "TORNTPHARM", "TVSMOTOR", "VEDL", "ZOMATO", "ZYDUSLIFE", "POLYCAB"]

def analyze_nse_stock(symbol, tf_interval, tf_period, label_str):
    try:
        df = yf.download(f"{symbol}.NS", period=tf_period, interval=tf_interval, progress=False)
        df = compute_indicators(df)
        if df is None or len(df) < 200: return None
        row = df.iloc[-1]
        
        # Logic
        close = round(row['close'], 2)
        high_125 = round(row['high_125'], 2)
        vol_ratio = round(row['volume'] / (row['vol_sma_125'] + 1e-9), 2)
        
        # New Filter Checks
        is_above_200sma = close > row['sma_200']
        is_vol_expanding = row['bb_width'] > df['bb_width'].shift(1).iloc[-1]
        
        is_buy = (close >= high_125) and (vol_ratio >= 1.0) and is_above_200sma and is_vol_expanding
        
        return {
            "Signal": "🟢 STRONG BUY" if is_buy else "NEUTRAL",
            "Stock": symbol,
            "Price": f"₹{close:,.2f}",
            "Score": "High" if is_above_200sma else "Low",
            "Vol Ratio": vol_ratio,
            "Is_Breakout": is_buy
        }
    except: return None

with main_tab2:
    st.title("🇮🇳 NSE Institutional Volume & Breakout Tracker")
    selected_tf = st.radio("Select Strategy Timeframe:", ("Daily (1D)", "1-Hour (1H)"), horizontal=True)
    interval = "1d" if "Daily" in selected_tf else "1h"
    period = "2y" if "Daily" in selected_tf else "60d"
    
    if st.button("Force Re-Scan NSE"): st.rerun()
    
    with st.spinner("Analyzing stocks with Institutional Filters..."):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(analyze_nse_stock, sym, interval, period, "125"): sym for sym in NIFTY_UNIVERSE}
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
            all_results = [r for r in results if r is not None]

    st.subheader("Active Institutional Breakouts (Trend + Volatility Confirmed)")
    df_results = pd.DataFrame([r for r in all_results if r['Is_Breakout']])
    if not df_results.empty:
        st.dataframe(df_results, use_container_width=True)
    else:
        st.info("No stocks currently meet strict Institutional Trend + Volatility criteria.")

# app.py.complete
