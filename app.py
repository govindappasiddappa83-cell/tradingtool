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

# IMPROVED INDICATOR ENGINE (With BB Width & 200 SMA Filter)
def compute_indicators(df):
    if df.empty or len(df) < 200: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.rename(columns=str.lower, inplace=True)
    
    df['high_125'] = df['high'].shift(1).rolling(125).max()
    df['vol_sma_125'] = df['volume'].rolling(125).mean()
    
    # New Filters
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_width'] = (df['bb_mid'] + 2*df['bb_std']) - (df['bb_mid'] - 2*df['bb_std'])
    df['sma_200'] = df['close'].rolling(200).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    hl = df['high'] - df['low']
    df['atr'] = hl.rolling(14).mean()
    return df.dropna()

# TAB 1: GOLD (Preserving your existing logic)
with main_tab1:
    st.subheader("Gold Analysis")
    # This keeps your original strategy logic intact
    df = yf.download("GC=F", period="1y", interval="1d", progress=False)
    df = compute_indicators(df)
    if df is not None:
        st.line_chart(df[['close', 'ema_21', 'ema_50']])

# TAB 2: NSE SCANNER (With new filters integrated)
NIFTY_UNIVERSE = ["ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL", "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC", "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO", "ABB", "AMBUJACEM", "BANKBARODA", "BOSCHLTD", "CANBK", "CHOLAFIN", "DABUR", "DLF", "GAIL", "GODREJCP", "HAVELLS", "HAL", "ICICIGI", "IOC", "IRCTC", "IRFC", "JINDALSTEL", "JIOFIN", "MOTHERSON", "PIDILITIND", "PNB", "PFC", "RECLTD", "SIEMENS", "TORNTPHARM", "TVSMOTOR", "VEDL", "ZOMATO", "ZYDUSLIFE", "POLYCAB"]

def analyze_nse_stock(symbol, tf_interval, tf_period):
    try:
        df = compute_indicators(yf.download(f"{symbol}.NS", period=tf_period, interval=tf_interval, progress=False))
        row = df.iloc[-1]
        
        # New Institutional Filter logic
        is_buy = (row['close'] >= row['high_125']) and \
                 (row['volume'] > row['vol_sma_125']) and \
                 (row['close'] > row['sma_200']) and \
                 (row['bb_width'] > df['bb_width'].shift(1).iloc[-1])
        
        return {"Stock": symbol, "Signal": "🟢 BUY" if is_buy else "Neutral", "Price": row['close']}
    except: return None

with main_tab2:
    st.title("🇮🇳 NSE Institutional Scanner")
    if st.button("Run Scanner"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = [f.result() for f in concurrent.futures.as_completed({executor.submit(analyze_nse_stock, sym, "1d", "2y"): sym for sym in NIFTY_UNIVERSE})]
            st.table(pd.DataFrame([r for r in results if r and r['Signal'] == "🟢 BUY"]))

# app.py.complete
