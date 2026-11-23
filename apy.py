import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

# Page Config
st.set_page_config(page_title='Dr.Gold AI', page_icon='🟡', layout='wide')
IST = ZoneInfo('Asia/Kolkata')
ENGINE_NAMES = [
    'Trend Following', 'Trend Pullback', 'Donchian Breakout', 'Mean Reversion',
    'Liquidity Sweep', 'Momentum', 'EMA Alignment', 'Volatility Breakout',
    'VWAP Context', 'Relative Volume'
]
TIMEFRAME_CONFIG = {
    '3 Min': {'interval': '1m', 'period': '7d', 'resample': '3min', 'minutes': 3},
    '15 Min': {'interval': '15m', 'period': '60d', 'resample': None, 'minutes': 15},
    '1 Hour': {'interval': '1h', 'period': '730d', 'resample': None, 'minutes': 60},
    '4 Hour': {'interval': '1h', 'period': '730d', 'resample': '4h', 'minutes': 240},
    'Monthly': {'interval': '1d', 'period': '10y', 'resample': 'MS', 'minutes': 43200},
}

# CSS Styles
st.markdown('''
<style>
.stApp {background:#f4f7fb !important; color:#172033 !important;}
[data-testid="stMetric"] {background:#ffffff !important; border:1px solid #d8e1ed !important; padding:14px !important; border-radius:12px !important;}
.buybox {background:#e7f7ef; border:1px solid #78c99e; color:#14532d; padding:17px; border-radius:12px;}
.sellbox {background:#fff0f0; border:1px solid #e9a1a1; color:#8f1d1d; padding:17px; border-radius:12px;}
.waitbox {background:#edf2f8; border:1px solid #bdc9d8; color:#26364a; padding:17px; border-radius:12px;}
</style>
''', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title('Dr.Gold AI settings')
    timeframe = st.selectbox('Analysis timeframe', list(TIMEFRAME_CONFIG), index=1)
    symbol_choice = st.selectbox('Public gold data symbol', ['XAUUSD=X', 'GC=F'], index=0)
    refresh_seconds = st.select_slider('Refresh attempt (seconds)', options=[30, 60, 120], value=30)
    if st.button('↻ Refresh data now'):
        st.cache_data.clear()
        st.rerun()

st_autorefresh(interval=refresh_seconds * 1000, key='drgold_refresh')

# Logic Functions
@st.cache_data(ttl=30)
def fetch_data(symbol, cfg):
    df = yf.download(symbol, period=cfg['period'], interval=cfg['interval'], progress=False)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
    if cfg['resample']:
        df = df.resample(cfg['resample']).agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'})
    return df.dropna()

def add_tech(df):
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    return df

# App Execution
try:
    cfg = TIMEFRAME_CONFIG[timeframe]
    data = fetch_data(symbol_choice, cfg)
    data = add_tech(data)
    
    st.title('🟡 Dr.Gold AI Terminal')
    
    # Decision Logic
    last = data.iloc[-1]
    decision = "BUY" if last['close'] > last['ema21'] else "SELL"
    css_class = "buybox" if decision == "BUY" else "sellbox"
    
    st.markdown(f'<div class="{css_class}"><h1>Decision: {decision}</h1></div>', unsafe_allow_html=True)
    
    st.line_chart(data[['close', 'ema21', 'ema50']])

    # Macro Section
    st.divider()
    st.subheader('CPI and global macro intelligence')
    st.info('Macro news and CPI actual/forecast monitoring remain a separate module to be integrated after the market feed is stable. They will not influence the technical ensemble until separately tested.')
    st.caption('Research tool only. No broker connection, order placement or guaranteed trading performance.')

except Exception as e:
    st.error(f"Waiting for market data... {e}")

# app.py.complete
