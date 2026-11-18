import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime
import pytz
import time

st.set_page_config(page_title="Professional Trading Dashboard", layout="wide", page_icon="⚡")

IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

# Top Navigation Tabs
main_tab1, main_tab2, main_tab3 = st.tabs([
    "🥇 Gold (XAU/USD) Dashboard", 
    "🇮🇳 NSE Stocks Scanner",
    "🎯 Intraday Option Momentum & Volatility Extension System"
])

# ========================================================
# HELPER: DATA CLEANER & INDICATOR ENGINE
# ========================================================
def clean_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = [str(c).lower() for c in df.columns.get_level_values(0)]
        if 'close' in lvl0 or 'open' in lvl0:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)
    df.columns = [str(c).lower() for c in df.columns]
    return df

def compute_indicators(df):
    if df is None or df.empty or len(df) < 130:
        return None

    df = clean_yfinance_columns(df)
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        return None

    df = df[required].copy()

    # 1. 125-Period Breakout Bounds (Shifted to eliminate lookahead bias)
    df['high_125'] = df['high'].shift(1).rolling(125).max()
    df['low_125'] = df['low'].shift(1).rolling(125).min()
    df['vol_sma_125'] = df['volume'].rolling(125).mean()

    # 2. RSI 14 (Wilder's RMA smoothing)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

    # 3. ATR 14 & ATR SMA 50
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['atr_sma_50'] = df['atr'].rolling(50).mean()

    # 4. Moving Averages
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(min(200, len(df)-1)).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    return df.dropna(subset=['high_125', 'vol_sma_125', 'rsi', 'atr'])

# ========================================================
# TAB 1: GOLD (XAU/USD) DASHBOARD
# ========================================================
with main_tab1:
    gold_strategy = st.radio(
        "🎯 **Choose Gold Strategy:**",
        ("Strategy 2: 30M Session VWAP + 9 EMA + PDH/PDL", "Strategy 1: 125-Bar Breakout + Volume (Multi-Timeframe)"),
        horizontal=True
    )

    if "Strategy 2" in gold_strategy:
        st.subheader("⚡ Gold Strategy 2: 30-Minute Session VWAP + 9 EMA + PDH / PDL")
        if st.button("🔄 Refresh 30M Gold Strategy", type="primary"):
            st.cache_data.clear()
            st.rerun()

        @st.cache_data(ttl=120)
        def fetch_gold_30m():
            df = yf.download("GC=F", period="30d", interval="30m", progress=False)
            if df.empty or len(df) < 50:
                return None, None, None

            df = clean_yfinance_columns(df)
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()

            df['date'] = df.index.date
            tp = (df['high'] + df['low'] + df['close']) / 3.0
            df['tp_vol'] = tp * df['volume']
            df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
            df['cum_vol'] = df.groupby('date')['volume'].cumsum()
            df['vwap'] = df['cum_tp_vol'] / (df['cum_vol'] + 1e-9)

            hl = df['high'] - df['low']
            hc = (df['high'] - df['close'].shift(1)).abs()
            lc = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

            counts = df.groupby('date').size()
            valid_days = counts[counts >= 10].index
            if len(valid_days) >= 2:
                prev_df = df[df['date'] == valid_days[-2]]
                pdh, pdl = float(prev_df['high'].max()), float(prev_df['low'].min())
            else:
                pdh, pdl = float(df['high'].max()), float(df['low'].min())

            return df.dropna(subset=['ema_9', 'vwap', 'atr']), pdh, pdl

        df_30m, pdh_val, pdl_val = fetch_gold_30m()

        if df_30m is not None and len(df_30m) >= 2:
            latest = df_30m.iloc[-1]
            p = round(float(latest['close']), 2)
            v = round(float(latest['vwap']), 2)
            e9 = round(float(latest['ema_9']), 2)
            a_val = round(float(latest['atr']), 2)

            is_bull = (p > v) and (e9 > v)
            is_bear = (p < v) and (e9 < v)

            if is_bull:
                sig = "<span style='background-color:#004D20; color:#00FF66; padding:6px 14px; border-radius:6px; font-weight:bold;'>🟢 STRONG BUY</span>"
                sl = f"${min(v - 1.0, round(p - 1.5 * a_val, 2)):,.2f}"
                tgt = f"${round(p + 3.0 * a_val, 2):,.2f}"
            elif is_bear:
                sig = "<span style='background-color:#4D0000; color:#FF4D4D; padding:6px 14px; border-radius:6px; font-weight:bold;'>🔴 STRONG SELL</span>"
                sl = f"${max(v + 1.0, round(p + 1.5 * a_val, 2)):,.2f}"
                tgt = f"${round(p - 3.0 * a_val, 2):,.2f}"
            else:
                sig = "<span style='background-color:#2A2A2A; color:#BBBBBB; padding:6px 14px; border-radius:6px; font-weight:bold;'>⚪ NEUTRAL</span>"
                sl, tgt = "-", "-"

            m1, m2, m3, m4 = st.columns([1.2, 2.2, 1.3, 1.3])
            m1.metric("Live Gold Price (30M)", f"${p:,.2f}")
            m2.markdown(f"**Signal**<br>{sig}", unsafe_allow_html=True)
            m3.metric("Stop-Loss", sl)
            m4.metric("Target (1:2 RR)", tgt)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Session VWAP", f"${v:,.2f}")
            c2.metric("9 EMA", f"${e9:,.2f}")
            c3.metric("PDH", f"${pdh_val:,.2f}")
            c4.metric("PDL", f"${pdl_val:,.2f}")

            plot_df = df_30m.iloc[-150:]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name="Gold"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['vwap'], line=dict(color='orange', width=2), name="VWAP"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_9'], line=dict(color='cyan', width=1.5), name="9 EMA"), row=1, col=1)
            fig.add_hline(y=pdh_val, line=dict(color='#00FF66', width=1.5, dash='dash'), annotation_text="PDH", row=1, col=1)
            fig.add_hline(y=pdl_val, line=dict(color='#FF4D4D', width=1.5, dash='dash'), annotation_text="PDL", row=1, col=1)
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], name="Volume"), row=2, col=1)
            fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("🏆 Gold Strategy 1: 125-Bar Breakout + Volume")
        gold_tf = st.radio("⏱️ Timeframe:", ("Daily (1D)", "4-Hour (4H)", "1-Hour (1H)", "15-Minute (15M)"), horizontal=True)
        tf_map = {
            "Daily (1D)": ("1d", "2y", "125-Day"),
            "4-Hour (4H)": ("4h", "120d", "125-Bar (4H)"),
            "1-Hour (1H)": ("1h", "60d", "125-Hour"),
            "15-Minute (15M)": ("15m", "30d", "125-Bar (15M)")
        }
        g_int, g_per, g_lbl = tf_map[gold_tf]

        @st.cache_data(ttl=120)
        def fetch_gold(interval_val, period_val):
            if interval_val == "4h":
                raw = yf.download("GC=F", period=period_val, interval="1h", progress=False)
                if raw.empty: return None
                raw = clean_yfinance_columns(raw)
                df = raw.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            else:
                df = yf.download("GC=F", period=period_val, interval=interval_val, progress=False)
            return compute_indicators(df)

        gold_df = fetch_gold(g_int, g_per)
        if gold_df is not None and len(gold_df) >= 2:
            row = gold_df.iloc[-1]
            p = round(float(row['close']), 2)
            h125 = round(float(row['high_125']), 2)
            a = round(float(row['atr']), 2)
            vr = round(float(row['volume']) / (float(row['vol_sma_125']) + 1e-9), 2)

            is_buy = (p >= h125) and (vr >= 1.0)
            is_sell = row['ema_21'] < row['ema_50']

            if is_buy:
                sig = "<span style='background-color:#004D20; color:#00FF66; padding:6px 14px; border-radius:6px; font-weight:bold;'>🟢 STRONG BUY</span>"
                sl = f"${round(p - 2 * a, 2):,.2f}"
                tgt = f"${round(p + 4 * a, 2):,.2f}"
            elif is_sell:
                sig = "<span style='background-color:#4D0000; color:#FF4D4D; padding:6px 14px; border-radius:6px; font-weight:bold;'>🔴 SELL / SHORT</span>"
                sl = f"${round(p + 2 * a, 2):,.2f}"
                tgt = f"${round(p - 4 * a, 2):,.2f}"
            else:
                sig = "<span style='background-color:#2A2A2A; color:#BBBBBB; padding:6px 14px; border-radius:6px; font-weight:bold;'>⚪ NEUTRAL</span>"
                sl, tgt = "-", "-"

            m1, m2, m3, m4 = st.columns([1.2, 2.2, 1.3, 1.3])
            m1.metric("Live Gold Price", f"${p:,.2f}")
            m2.markdown(f"**Signal**<br>{sig}", unsafe_allow_html=True)
            m3.metric("Stop-Loss (2 ATR)", sl)
            m4.metric("Target (4 ATR)", tgt)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=gold_df.index, open=gold_df['open'], high=gold_df['high'], low=gold_df['low'], close=gold_df['close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['high_125'], line=dict(color='orange', dash='dash'), name=f"{g_lbl} High"), row=1, col=1)
            fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['ema_21'], line=dict(color='blue', width=1.5), name="21 EMA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['ema_50'], line=dict(color='red', width=1.5), name="50 EMA"), row=1, col=1)
            fig.add_trace(go.Bar(x=gold_df.index, y=gold_df['volume'], name="Volume"), row=2, col=1)
            fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

# ========================================================
# TAB 2: NSE STOCKS SCANNER
# ========================================================
with main_tab2:
    st.title("🇮🇳 NSE Institutional Volume & Breakout Tracker")

    selected_tf = st.radio("⏱️ Select Timeframe:", ("📅 Daily (1D) — Positional Swing", "⏰ 1-Hour (1H) — Short-Term Swing"), horizontal=True)
    is_daily = "Daily" in selected_tf
    interval = "1d" if is_daily else "1h"
    period = "2y" if is_daily else "60d"
    bar_label = "125-Day" if is_daily else "125-Hour"

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

    def analyze_nse_stock(symbol, tf_int, tf_per):
        ticker = f"{symbol}.NS"
        df = None
        for _ in range(2):
            try:
                df = yf.download(ticker, period=tf_per, interval=tf_int, progress=False)
                if df is not None and not df.empty and len(df) >= 130:
                    break
            except Exception:
                time.sleep(0.2)
                continue

        df = compute_indicators(df)
        if df is None or len(df) < 2: return None

        try:
            row = df.iloc[-1]
            prev = df.iloc[-2]
            p = round(float(row['close']), 2)
            h125 = round(float(row['high_125']), 2)
            a = round(float(row['atr']), 2)
            vr = round(float(row['volume']) / (float(row['vol_sma_125']) + 1e-9), 2)
            dist = round(((p - h125) / h125) * 100, 2)

            is_bo = (p >= h125) and (vr >= 1.0)
            is_sell = (row['ema_21'] < row['ema_50']) and (prev['ema_21'] >= prev['ema_50'])
            is_near = (-3.0 <= dist < 0.0) and (vr >= 0.8)

            score = sum([int(row['rsi'] < 70), int(row['atr'] > row['atr_sma_50']), int(row['close'] > row['sma_50']), int(row['ema_21'] > row['ema_50'])])

            return {
                "Signal": "🟢 BUY" if is_bo else ("🔴 SELL" if is_sell else ("🟡 NEAR BUY" if is_near else "⚪ NEUTRAL")),
                "Stock": symbol,
                "Price": f"₹{p:,.2f}",
                "Breakout High": f"₹{h125:,.2f}",
                "Distance to High": f"{dist:+0.2f}%",
                "Volume Surge": f"{vr}x Avg",
                "RSI": round(float(row['rsi']), 1),
                "Score": f"{score + (1 if is_bo else 0)}/6",
                "Stop Loss": f"₹{round(p - 2 * a, 2):,.2f}",
                "Target": f"₹{round(p + 4 * a, 2):,.2f}",
                "Is_Breakout": is_bo,
                "Is_High_Volume": vr >= 1.2,
                "Is_Near_Breakout": is_near,
                "Is_Sell": is_sell,
                "Raw_Volume": vr,
                "Raw_Distance": dist
            }
        except Exception:
            return None

    @st.cache_data(ttl=600)
    def run_clean_scan(tf_int, tf_per):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(analyze_nse_stock, sym, tf_int, tf_per): sym for sym in NIFTY_UNIVERSE}
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        return [r for r in results if r is not None]

    if st.button("🔄 Force Re-Scan NSE", type="primary"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Screening NSE stocks..."):
        all_results = run_clean_scan(interval, period)

    # 1. Breakouts
    st.subheader(f"🎯 1. Active Breakouts ({bar_label} High + Volume)")
    bo = [r for r in all_results if r.get("Is_Breakout")]
    if bo:
        st.dataframe(pd.DataFrame(bo)[["Signal", "Stock", "Price", "Breakout High", "Volume Surge", "Score", "Stop Loss", "Target", "RSI"]], use_container_width=True)
    else:
        st.info("No stocks currently meeting active breakout rules.")

    # 2. Top Volume
    st.subheader("⚡ 2. Top Volume Movers Today")
    vm = sorted([r for r in all_results if r.get("Is_High_Volume")], key=lambda x: x.get("Raw_Volume", 0), reverse=True)
    if vm:
        st.dataframe(pd.DataFrame(vm)[["Signal", "Stock", "Price", "Volume Surge", "Distance to High", "RSI", "Score"]], use_container_width=True)

    # 3. Near Breakout
    st.subheader(f"🔭 3. Near-Breakout Radar (Within 3%)")
    nr = sorted([r for r in all_results if r.get("Is_Near_Breakout")], key=lambda x: x.get("Raw_Distance", 0), reverse=True)
    if nr:
        st.dataframe(pd.DataFrame(nr)[["Signal", "Stock", "Price", "Breakout High", "Distance to High", "Volume Surge", "RSI", "Score"]], use_container_width=True)

    # 4. Exits
    st.subheader("🔴 4. Trend Exit Alerts (21 EMA < 50 EMA)")
    ex = [r for r in all_results if r.get("Is_Sell")]
    if ex:
        st.dataframe(pd.DataFrame(ex)[["Signal", "Stock", "Price", "RSI"]], use_container_width=True)

# ========================================================
# TAB 3: INTRADAY OPTION MOMENTUM & VOLATILITY EXTENSION SYSTEM
# ========================================================
with main_tab3:
    st.header("🎯 Intraday Option Momentum & Volatility Extension System")
    st.caption("Custom Volatility Extension Ladder, Move View Fail Triggers & Multi-Tier Range Targets")

    col_c1, col_c2, col_c3 = st.columns([1.5, 1.5, 1.2])
    with col_c1:
        user_ticker = st.text_input("Enter Ticker / Symbol:", value="^NSEI", help="e.g., ^NSEI, ^NSEBANK, RELIANCE.NS, GC=F").strip().upper()
    with col_c2:
        selected_intraday_tf = st.selectbox("Select Timeframe:", ("3m (Fast Momentum)", "5m (Standard Options)", "15m (Trend Swing)", "1m (Scalp)"), index=1)
    with col_c3:
        st.write("")
        st.write("")
        if st.button("🔄 Recalculate Model", type="primary"):
            st.cache_data.clear()
            st.rerun()

    opt_tf_map = {
        "1m (Scalp)": ("1m", "5d"),
        "3m (Fast Momentum)": ("3m", "5d"),
        "5m (Standard Options)": ("5m", "10d"),
        "15m (Trend Swing)": ("15m", "30d")
    }
    opt_interval, opt_period = opt_tf_map[selected_intraday_tf]

    @st.cache_data(ttl=60)
    def fetch_volatility_extension_data(symbol, tf_int, tf_per):
        try:
            df = yf.download(symbol, period=tf_per, interval=tf_int, progress=False)
            if df.empty or len(df) < 30: return None
            df = clean_yfinance_columns(df)
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()

            hl = df['high'] - df['low']
            hc = (df['high'] - df['close'].shift(1)).abs()
            lc = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            return df.dropna(subset=['atr', 'ema_9', 'ema_21'])
        except Exception:
            return None

    with st.spinner("Calculating Volatility Extension Ladders..."):
        v_df = fetch_volatility_extension_data(user_ticker, opt_interval, opt_period)

    if v_df is None or len(v_df) < 5:
        st.error(f"Could not load data for '{user_ticker}'. Please verify the symbol.")
    else:
        last = v_df.iloc[-1]
        c_price = float(last['close'])
        c_atr = float(last['atr'])
        c_ema9 = float(last['ema_9'])
        c_ema21 = float(last['ema_21'])

        is_bullish = (c_price >= c_ema9) and (c_ema9 >= c_ema21)
        is_bearish = (c_price < c_ema9) and (c_ema9 < c_ema21)

        risk_step = round(c_atr * 1.5, 2)
        half_step = round(risk_step / 2.0, 2)

        if is_bullish:
            bias_title = "CONTINUE VIEW UP"
            box_bg = "#113b22"
            box_border = "#00FF66"
            fail_lvl = round(c_price - risk_step, 2)
            fail_str = f"{fail_lvl:,.2f} ({risk_step:,.2f} pts)"

            r1 = round(c_price + risk_step, 2)
            r2 = round(r1 + half_step, 2)
            r3 = round(r2 + half_step, 2)
            r4 = round(r3 + half_step, 2)
            r5 = round(r4 + half_step, 2)

            ladder_labels = [
                ("N Upper Range 1", r1),
                ("N Upper Range 2", r2),
                ("N Upper Range 3", r3),
                ("N Extended Upper Range 4", r4),
                ("N Extended Upper Range 5", r5)
            ]
        elif is_bearish:
            bias_title = "CONTINUE VIEW DOWN"
            box_bg = "#421616"
            box_border = "#FF4D4D"
            fail_lvl = round(c_price + risk_step, 2)
            fail_str = f"{fail_lvl:,.2f} ({risk_step:,.2f} pts)"

            r1 = round(c_price - risk_step, 2)
            r2 = round(r1 - half_step, 2)
            r3 = round(r2 - half_step, 2)
            r4 = round(r3 - half_step, 2)
            r5 = round(r4 - half_step, 2)

            ladder_labels = [
                ("N Lower Range 1", r1),
                ("N Lower Range 2", r2),
                ("N Lower Range 3", r3),
                ("N Extended Lower Range 4", r4),
                ("N Extended Lower Range 5", r5)
            ]
        else:
            bias_title = "NEUTRAL / CONSOLIDATION"
            box_bg = "#222222"
            box_border = "#888888"
            fail_lvl = round(c_price - risk_step, 2)
            fail_str = f"{fail_lvl:,.2f} ({risk_step:,.2f} pts)"

            r1 = round(c_price + risk_step, 2)
            r2 = round(r1 + half_step, 2)
            r3 = round(r2 + half_step, 2)
            r4 = round(r3 + half_step, 2)
            r5 = round(r4 + half_step, 2)

            ladder_labels = [
                ("Range Level 1", r1),
                ("Range Level 2", r2),
                ("Range Level 3", r3),
                ("Extended Range 4", r4),
                ("Extended Range 5", r5)
            ]

        # Display Signal Box
        time_str = now_ist.strftime("%d %b %I:%M %p (IST)")
        box_html = f"""
        <div style="background-color:{box_bg}; border:2px solid {box_border}; border-radius:10px; padding:18px; margin-bottom:20px;">
            <h3 style="color:#FFFFFF; margin-top:0;">{bias_title}</h3>
            <p style="color:#E0E0E0; margin-bottom:8px; font-size:14px;">📅 {time_str}</p>
            <div style="background-color:rgba(0,0,0,0.3); padding:10px; border-radius:6px; margin-bottom:12px;">
                <span style="font-weight:bold; color:#FFAA00;">Move View Fail LVL:</span> 
                <span style="color:#FFFFFF; font-weight:bold; font-size:16px;">{fail_str}</span>
            </div>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:10px;">
        """
        for name, lvl in ladder_labels:
            box_html += f"""
                <div style="background-color:rgba(255,255,255,0.06); padding:8px 12px; border-radius:6px;">
                    <div style="color:#B0B0B0; font-size:12px;">{name}</div>
                    <div style="color:#FFFFFF; font-weight:bold; font-size:16px;">{lvl:,.2f}</div>
                </div>
            """
        box_html += "</div></div>"
        st.markdown(box_html, unsafe_allow_html=True)

        # Plotly Visual
        st.subheader(f"📊 Live Intraday Chart with Target Ladder ({user_ticker})")
        p_df = v_df.iloc[-90:]

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['open'], high=p_df['high'], low=p_df['low'], close=p_df['close'], name=user_ticker))
        fig.add_trace(go.Scatter(x=p_df.index, y=p_df['ema_9'], line=dict(color='cyan', width=1.5), name="9 EMA"))
        fig.add_trace(go.Scatter(x=p_df.index, y=p_df['ema_21'], line=dict(color='magenta', width=1.5), name="21 EMA"))

        fig.add_hline(y=fail_lvl, line=dict(color='#FFAA00', width=2, dash='dot'), annotation_text=f"Fail LVL: {fail_lvl:,.2f}")
        colors = ['#66FF66', '#33CC33', '#009900', '#006600', '#004400']
        for i, (name, lvl) in enumerate(ladder_labels):
            fig.add_hline(y=lvl, line=dict(color=colors[i % len(colors)], width=1, dash='dash'), annotation_text=f"{name}: {lvl:,.2f}")

        fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=30, b=30, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)
