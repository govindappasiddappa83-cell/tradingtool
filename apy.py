import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime, time
import pytz
from kiteconnect import KiteConnect

st.set_page_config(page_title="NSE Signals & After-Market Execution", layout="wide", page_icon="⚡")

# Timezone check for Indian Market
IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)
market_open = time(9, 15)
market_close = time(15, 30)
current_time = now_ist.time()

# Determine if market is currently live or in After-Market hours
is_market_live = (now_ist.weekday() < 5) and (market_open <= current_time <= market_close)

st.title("⚡ NSE Actionable Signals & Execution Dashboard")
if is_market_live:
    st.success(f"🟢 **MARKET IS LIVE** | Current Time (IST): {now_ist.strftime('%I:%M %p')}")
else:
    st.info(f"🌙 **AFTER-MARKET HOURS (Next-Day Planning)** | Orders will be placed as **AMO** for the 9:15 AM open | Current Time: {now_ist.strftime('%d-%b %I:%M %p')}")

# ========================================================
# 1. ZERODHA AUTHENTICATION (SIDEBAR)
# ========================================================
st.sidebar.header("🔐 Zerodha Connection")

api_key = st.secrets.get("KITE_API_KEY", "")
api_secret = st.secrets.get("KITE_API_SECRET", "")

kite = None
if "kite_session" not in st.session_state:
    st.session_state["kite_session"] = None

if api_key and api_secret:
    kite = KiteConnect(api_key=api_key)
    query_params = st.query_params
    if "request_token" in query_params:
        try:
            session_data = kite.generate_session(query_params["request_token"], api_secret=api_secret)
            st.session_state["kite_session"] = session_data["access_token"]
            st.sidebar.success("✅ Connected to Zerodha!")
        except Exception as e:
            st.sidebar.error(f"Login failed: {e}")

    if st.session_state["kite_session"]:
        kite.set_access_token(st.session_state["kite_session"])
        try:
            profile = kite.profile()
            margins = kite.margins(segment="equity")
            cash = margins.get("net", 0.0)
            st.sidebar.success(f"User: **{profile['user_name']}**")
            st.sidebar.metric("Available Margin", f"₹{cash:,.2f}")
        except:
            st.session_state["kite_session"] = None
    else:
        st.sidebar.link_button("🔑 Login to Zerodha", kite.login_url(), type="primary")

# ========================================================
# 2. STRATEGY ENGINE (Rules A through K)
# ========================================================
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
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO"
]

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

        # 125D High of previous sessions
        df['high_125'] = df['high'].shift(1).rolling(125).max()
        df['vol_sma_125'] = df['volume'].rolling(125).mean()

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

        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
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

        vol_surge_ratio = round(row['volume'] / (row['vol_sma_125'] + 1e-9), 2)
        score = sum([c_a, c_b, c_c, c_d, c_hi, c_j])

        close = round(row['close'], 2)
        atr_val = round(row['atr'], 2)

        if score == 6:
            return {
                "Type": "BUY",
                "Stock": symbol,
                "Price": close,
                "StopLoss": round(close - (2 * atr_val), 2),
                "Target": round(close + (4 * atr_val), 2),
                "VolumeSurge": f"{vol_surge_ratio}x Avg",
                "RSI": round(row['rsi'], 1)
            }
        elif is_sell:
            return {
                "Type": "SELL",
                "Stock": symbol,
                "Price": close,
                "Reason": "21 EMA crossed below 50 EMA"
            }
        return None
    except:
        return None

@st.cache_data(ttl=900)
def run_scan():
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_stock, NIFTY_UNIVERSE))
    buys = [r for r in results if r and r["Type"] == "BUY"]
    sells = [r for r in results if r and r["Type"] == "SELL"]
    return buys, sells

# ========================================================
# 3. DISPLAY & SMART ORDER PLACEMENT
# ========================================================
col_top1, col_top2 = st.columns([3, 1])
with col_top2:
    if st.button("🔄 Re-Scan Now", type="primary"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Analyzing closed daily candles and volume surges..."):
    buys, sells = run_scan()

st.subheader("🟢 Confirmed BUY Setups for Entry")
if buys:
    for b in buys:
        with st.container():
            c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1.2, 1.2, 1.2, 1.2, 2.2])
            c1.markdown(f"### **{b['Stock']}**")
            c2.metric("Closing Price", f"₹{b['Price']}")
            c3.metric("Stop Loss", f"₹{b['StopLoss']}")
            c4.metric("Target", f"₹{b['Target']}")
            c5.metric("Volume Surge", b['VolumeSurge'])

            with c6:
                qty = st.number_input(f"Shares ({b['Stock']})", min_value=1, value=1, key=f"qty_{b['Stock']}")
                btn_label = "⚡ Buy Live (Market)" if is_market_live else "🌙 Place AMO (For Next Open)"
                
                if st.button(btn_label, key=f"btn_{b['Stock']}"):
                    if st.session_state.get("kite_session") and kite:
                        try:
                            # Automatically choose Variety: REGULAR (if live) vs AMO (if after hours)
                            order_variety = kite.VARIETY_REGULAR if is_market_live else kite.VARIETY_AMO

                            # 1. Place Buy Order
                            order_id = kite.place_order(
                                variety=order_variety,
                                exchange=kite.EXCHANGE_NSE,
                                tradingsymbol=b['Stock'],
                                transaction_type=kite.TRANSACTION_TYPE_BUY,
                                quantity=qty,
                                order_type=kite.ORDER_TYPE_MARKET,
                                product=kite.PRODUCT_CNC
                            )

                            if is_market_live:
                                # Stop-loss limit trigger during live market
                                kite.place_order(
                                    variety=kite.VARIETY_REGULAR,
                                    exchange=kite.EXCHANGE_NSE,
                                    tradingsymbol=b['Stock'],
                                    transaction_type=kite.TRANSACTION_TYPE_SELL,
                                    quantity=qty,
                                    order_type=kite.ORDER_TYPE_SL_M,
                                    trigger_price=b['StopLoss'],
                                    product=kite.PRODUCT_CNC
                                )
                                st.success(f"✅ Live Order Placed! ID: {order_id}. Stop-Loss trigger set at ₹{b['StopLoss']}.")
                            else:
                                st.success(f"✅ AMO Queued with Zerodha! ID: {order_id}. It will execute at 9:15 AM tomorrow. (Remember to place your SL trigger of ₹{b['StopLoss']} once executed).")
                        except Exception as e:
                            st.error(f"Zerodha Order Failed: {e}")
                    else:
                        st.warning("👉 Please log in to Zerodha in the left sidebar first!")
            st.divider()
else:
    st.info("ℹ️ No stocks passed all 6 rules on today's closing candle. Your capital stays safely in cash.")

st.subheader("🔴 Confirmed SELL / Trend Exit Signals")
if sells:
    st.dataframe(pd.DataFrame(sells).drop(columns=["Type"]), use_container_width=True)
else:
    st.info("ℹ️ No active sell signals.")
