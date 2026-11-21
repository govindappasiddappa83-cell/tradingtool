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

def wilder_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def wilder_atr(df, period=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def check_freshness(index, interval_minutes):
    last_ts = index[-1]
    if last_ts.tzinfo is None:
        last_ts = pytz.utc.localize(last_ts)
    age_min = (datetime.now(pytz.utc) - last_ts.astimezone(pytz.utc)).total_seconds() / 60.0
    max_age = max(interval_minutes * 2, 30)
    return age_min, age_min <= max_age

def compute_indicators(df):
    if df is None or df.empty or len(df) < 130:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.rename(columns=str.lower, inplace=True)
    df['high_125'] = df['high'].shift(1).rolling(125).max()
    df['low_125'] = df['low'].shift(1).rolling(125).min()
    df['vol_sma_125'] = df['volume'].rolling(125).mean()
    df['rsi'] = wilder_rsi(df['close'], 14)
    df['atr'] = wilder_atr(df, 14)
    df['atr_sma_50'] = df['atr'].rolling(50).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(min(200, len(df) - 1)).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    return df.dropna(subset=['high_125', 'vol_sma_125', 'rsi', 'atr'])

with main_tab1:
    gold_strategy = st.radio(
        "🎯 **Choose Gold Strategy:**",
        ("Strategy 2: 30M Session VWAP + 9 EMA + PDH/PDL", "Strategy 1: 125-Bar Breakout + Volume (Multi-Timeframe)"),
        horizontal=True
    )

    if "Strategy 2" in gold_strategy:
        st.subheader("⚡ Gold Strategy 2: 30-Minute Session VWAP + 9 EMA + PDH / PDL")
        st.caption("Institutional intraday momentum: Session-reset VWAP, 9 EMA (Close), and Previous Day High/Low reference levels.")

        if st.button("🔄 Refresh 30M Gold Strategy", type="primary"):
            st.cache_data.clear()
            st.rerun()

        @st.cache_data(ttl=120)
        def fetch_gold_30m():
            df = yf.download("GC=F", period="30d", interval="30m", progress=False)
            if df.empty or len(df) < 50:
                return None, None, None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.rename(columns=str.lower, inplace=True)
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['date'] = df.index.date
            typical_price = (df['high'] + df['low'] + df['close']) / 3.0
            df['tp_vol'] = typical_price * df['volume']
            df['cum_tp_vol'] = df.groupby('date')['tp_vol'].cumsum()
            df['cum_vol'] = df.groupby('date')['volume'].cumsum()
            df['vwap'] = df['cum_tp_vol'] / (df['cum_vol'] + 1e-9)
            df['atr'] = wilder_atr(df, 14)
            daily_df = df.groupby('date').agg({'high': 'max', 'low': 'min'})
            if len(daily_df) >= 2:
                pdh = float(daily_df['high'].iloc[-2])
                pdl = float(daily_df['low'].iloc[-2])
            else:
                pdh = float(df['high'].max())
                pdl = float(df['low'].min())
            return df.dropna(subset=['ema_9', 'vwap', 'atr']), pdh, pdl

        with st.spinner("Calculating 30M Session VWAP and Previous Day levels..."):
            df_30m, pdh_val, pdl_val = fetch_gold_30m()

        if df_30m is None or len(df_30m) < 2:
            st.error("Could not fetch 30M Gold market data. Try refreshing.")
        else:
            age_min, is_fresh = check_freshness(df_30m.index, 30)
            if not is_fresh:
                st.warning(f"⚠️ Last candle is {age_min:.0f} min old — data may be stale.")

            latest = df_30m.iloc[-1]
            cur_price = round(latest['close'], 2)
            cur_vwap = round(latest['vwap'], 2)
            cur_ema9 = round(latest['ema_9'], 2)
            atr_30m = round(latest['atr'], 2)

            is_bullish = (cur_price > cur_vwap) and (cur_ema9 > cur_vwap)
            is_bearish = (cur_price < cur_vwap) and (cur_ema9 < cur_vwap)

            if cur_price > pdh_val:
                pd_status = "🚀 Above Previous Day High (PDH Breakout — Very Bullish)"
            elif cur_price < pdl_val:
                pd_status = "⚠️ Below Previous Day Low (PDL Breakdown — Very Bearish)"
            else:
                pd_status = "↔️ Trading Inside Previous Day Range"

            if is_bullish:
                sig_html = "<span style='background-color:#004D20; color:#00FF66; padding:6px 14px; border-radius:6px; font-weight:bold; font-size:20px;'>🟢 STRONG BUY (Price & 9 EMA Above VWAP)</span>"
                sl_label = "Stop-Loss (Below VWAP / -1.5 ATR)"
                sl_val = f"${min(cur_vwap - 1.0, round(cur_price - (1.5 * atr_30m), 2)):,.2f}"
                tgt_label = "Target (+3 ATR, 1:2 RR)"
                tgt_val = f"${round(cur_price + (3.0 * atr_30m), 2):,.2f}"
            elif is_bearish:
                sig_html = "<span style='background-color:#4D0000; color:#FF4D4D; padding:6px 14px; border-radius:6px; font-weight:bold; font-size:20px;'>🔴 STRONG SELL / SHORT (Price & 9 EMA Below VWAP)</span>"
                sl_label = "Stop-Loss (Above VWAP / +1.5 ATR)"
                sl_val = f"${max(cur_vwap + 1.0, round(cur_price + (1.5 * atr_30m), 2)):,.2f}"
                tgt_label = "Target (-3 ATR, 1:2 RR)"
                tgt_val = f"${round(cur_price - (3.0 * atr_30m), 2):,.2f}"
            else:
                sig_html = "<span style='background-color:#2A2A2A; color:#BBBBBB; padding:6px 14px; border-radius:6px; font-weight:bold; font-size:20px;'>⚪ NEUTRAL / CROSSING VWAP</span>"
                sl_label, sl_val, tgt_label, tgt_val = "Stop-Loss", "-", "Target", "-"

            m1, m2, m3, m4 = st.columns([1.2, 2.2, 1.3, 1.3])
            m1.metric("Live Gold Price (30M)", f"${cur_price:,.2f}")
            m2.markdown(f"**Signal**  \n<div style='margin-top:6px;'>{sig_html}</div>", unsafe_allow_html=True)
            m3.metric(sl_label, sl_val)
            m4.metric(tgt_label, tgt_val)

            st.divider()
            c_chk1, c_chk2, c_chk3, c_chk4 = st.columns(4)
            c_chk1.metric("Session VWAP", f"${cur_vwap:,.2f}")
            c_chk2.metric("9 EMA (Close)", f"${cur_ema9:,.2f}")
            c_chk3.metric("PDH (Prev Day High)", f"${pdh_val:,.2f}")
            c_chk4.metric("PDL (Prev Day Low)", f"${pdl_val:,.2f}")
            st.info(f"**PDH/PDL Context:** {pd_status}")
            st.divider()

            st.subheader("Gold 30-Minute Chart (Session VWAP + 9 EMA + PDH + PDL)")
            plot_df = df_30m.iloc[-150:]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name="Gold (30M)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['vwap'], line=dict(color='orange', width=2.5), name="Session VWAP"), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_9'], line=dict(color='cyan', width=1.8), name="9 EMA (Close)"), row=1, col=1)
            fig.add_hline(y=pdh_val, line=dict(color='#00FF66', width=2, dash='dash'), annotation_text=f"PDH: ${pdh_val:,.2f}", annotation_position="top right", row=1, col=1)
            fig.add_hline(y=pdl_val, line=dict(color='#FF4D4D', width=2, dash='dash'), annotation_text=f"PDL: ${pdl_val:,.2f}", annotation_position="bottom right", row=1, col=1)
            v_colors = ['green' if c >= o else 'red' for c, o in zip(plot_df['close'], plot_df['open'])]
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], marker_color=v_colors, name="COMEX Volume"), row=2, col=1)
            fig.update_layout(height=750, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("🥇 Gold Strategy 1: 125-Bar Breakout + Volume (Multi-Timeframe)")
        c_g1, c_g2 = st.columns([2, 1])
        with c_g1:
            gold_tf = st.radio(
                "⏱️ **Select Gold Timeframe:**",
                ("Daily (1D) — Macro Swing", "4-Hour (4H) — Swing", "1-Hour (1H) — Intraday / Short-Term", "15-Minute (15M) — Scalp"),
                horizontal=True
            )
        with c_g2:
            if st.button("🔄 Refresh Gold Price", type="primary"):
                st.cache_data.clear()
                st.rerun()

        tf_map = {
            "Daily (1D) — Macro Swing": ("1d", "2y", False, "125-Day", 1440),
            "4-Hour (4H) — Swing": ("1h", "120d", True, "125-Bar (4H)", 240),
            "1-Hour (1H) — Intraday / Short-Term": ("1h", "60d", False, "125-Hour", 60),
            "15-Minute (15M) — Scalp": ("15m", "30d", False, "125-Bar (15M)", 15)
        }
        g_interval, g_period, need_resample, g_bar_label, g_minutes = tf_map[gold_tf]

        @st.cache_data(ttl=120)
        def fetch_gold(interval_val, period_val, resample_flag):
            df = yf.download("GC=F", period=period_val, interval=interval_val, progress=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if resample_flag:
                df = df.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
            return compute_indicators(df)

        gold_df = fetch_gold(g_interval, g_period, need_resample)

        if gold_df is None or len(gold_df) < 2:
            st.error("Could not fetch sufficient Gold candles.")
        else:
            age_min, is_fresh = check_freshness(gold_df.index, g_minutes)
            if not is_fresh:
                st.warning(f"⚠️ Last candle is {age_min:.0f} min old — data may be stale for this timeframe.")

            row = gold_df.iloc[-1]
            close = round(row['close'], 2)
            high_125 = round(row['high_125'], 2)
            atr_val = round(row['atr'], 2)
            vol_ratio = round(row['volume'] / (row['vol_sma_125'] + 1e-9), 2)

            c_a = close >= high_125
            c_b = vol_ratio >= 1.0
            c_c = row['rsi'] < 70
            c_d = row['atr'] > row['atr_sma_50']
            c_e = row['close'] > row['sma_50']
            c_f = row['ema_21'] > row['ema_50']
            is_bearish_ema = row['ema_21'] < row['ema_50']
            score = sum([c_a, c_b, c_c, c_d, c_e, c_f])

            if c_a and c_b:
                sig_html = "<span style='background-color:#004D20; color:#00FF66; padding:6px 14px; border-radius:6px; font-weight:bold; font-size:20px;'>🟢 STRONG BUY</span>"
                sl_label = "Stop-Loss (-2 ATR, Below)"
                sl_val = f"${round(close - (2 * atr_val), 2):,.2f}"
                tgt_label = "Target (+4 ATR, 1:2 RR)"
                tgt_val = f"${round(close + (4 * atr_val), 2):,.2f}"
            elif is_bearish_ema:
                sig_html = "<span style='background-color:#4D0000; color:#FF4D4D; padding:6px 14px; border-radius:6px; font-weight:bold; font-size:20px;'>🔴 SELL / SHORT</span>"
                sl_label = "Stop-Loss (+2 ATR, Above)"
                sl_val = f"${round(close + (2 * atr_val), 2):,.2f}"
                tgt_label = "Target (-4 ATR, 1:2 RR)"
                tgt_val = f"${round(close - (4 * atr_val), 2):,.2f}"
            else:
                sig_html = "<span style='background-color:#2A2A2A; color:#BBBBBB; padding:6px 14px; border-radius:6px; font-weight:bold; font-size:20px;'>⚪ NEUTRAL</span>"
                sl_label, sl_val, tgt_label, tgt_val = "Stop-Loss", "-", "Target", "-"

            m1, m2, m3, m4 = st.columns([1.2, 2.2, 1.3, 1.3])
            m1.metric("Live Gold Price (USD)", f"${close:,.2f}")
            m2.markdown(f"**Signal**  \n<div style='margin-top:6px;'>{sig_html}</div>", unsafe_allow_html=True)
            m3.metric(sl_label, sl_val)
            m4.metric(tgt_label, tgt_val)

            st.metric("Confirmation Score (of 6 checks)", f"{score}/6")
            st.divider()

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=gold_df.index, open=gold_df['open'], high=gold_df['high'], low=gold_df['low'], close=gold_df['close'], name="Price"), row=1,
            fig.add_trace(go.Candlestick(x=gold_df.index, open=gold_df['open'], high=gold_df['high'], low=gold_df['low'], close=gold_df['close'], name="Price"), row=1, col=1)
            fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['high_125'], line=dict(color='orange', dash='dash'), name=f"{g_bar_label} High"), row=1, col=1)
            fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['ema_21'], line=dict(color='blue', width=1.5), name="21 EMA"), row=1, col=1)
            fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['ema_50'], line=dict(color='red', width=1.5), name="50 EMA"), row=1, col=1)
            v_colors = ['green' if c >= o else 'red' for c, o in zip(gold_df['close'], gold_df['open'])]
            fig.add_trace(go.Bar(x=gold_df.index, y=gold_df['volume'], marker_color=v_colors, name="Volume"), row=2, col=1)
            fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

with main_tab2:
    st.title("🇮🇳 NSE Institutional Volume & Breakout Tracker")

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
    period = "2y" if is_daily else "60d"
    bar_label = "125-Day" if is_daily else "125-Hour"
    tf_minutes = 1440 if is_daily else 60

    st.caption(f"Strategy: {bar_label} High + Volume Surge + RSI < 70 + Volatility + SMA 50/200 + EMA 21/50 | Updated: {now_ist.strftime('%d-%b-%Y %I:%M %p IST')}")

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

    def analyze_nse_stock(symbol, tf_interval, tf_period, label_str):
        try:
            ticker = f"{symbol}.NS"
            df = yf.download(ticker, period=tf_period, interval=tf_interval, progress=False)
            df = compute_indicators(df)
            if df is None or len(df) < 2:
                return None

            row = df.iloc[-1]
            prev = df.iloc[-2]

            close = round(row['close'], 2)
            high_125 = round(row['high_125'], 2)
            atr_val = round(row['atr'], 2)
            vol_ratio = round(row['volume'] / (row['vol_sma_125'] + 1e-9), 2)
            pct_from_high = round(((close - high_125) / high_125) * 100, 2)

            cond_a = close >= high_125
            cond_b = vol_ratio >= 1.0
            cond_c = row['rsi'] < 70
            cond_d = row['atr'] > row['atr_sma_50']
            cond_e = row['close'] > row['sma_50']
            cond_f = row['ema_21'] > row['ema_50']

            is_buy = cond_a and cond_b
            is_sell = row['ema_21'] < row['ema_50']
            is_near = (-3.0 <= pct_from_high < 0.0) and (vol_ratio >= 0.8)

            if is_buy:
                signal_label = "🟢 BUY"
            elif is_sell:
                signal_label = "🔴 SELL"
            elif is_near:
                signal_label = "🟡 NEAR BUY"
            else:
                signal_label = "⚪ NEUTRAL"

            confirm_score = sum([cond_c, cond_d, cond_e, cond_f])

            age_min, is_fresh = check_freshness(df.index, tf_minutes)

            return {
                "Signal": signal_label,
                "Stock": symbol,
                "Price": f"₹{close:,.2f}",
                "Breakout High": f"₹{high_125:,.2f}",
                "Distance to High": f"{pct_from_high:+0.2f}%",
                "Volume Surge": f"{vol_ratio}x Avg",
                "RSI": round(row['rsi'], 1),
                "Score": f"{confirm_score + (1 if cond_a else 0) + (1 if cond_b else 0)}/6",
                "Stop Loss": f"₹{round(close - (2 * atr_val), 2):,.2f}",
                "Target": f"₹{round(close + (4 * atr_val), 2):,.2f}",
                "Data Age (min)": round(age_min, 0),
                "Fresh": "🟢" if is_fresh else "🔴 Stale",
                "Is_Breakout": is_buy,
                "Is_High_Volume": vol_ratio >= 1.2,
                "Is_Near_Breakout": is_near,
                "Is_Sell": is_sell,
                "Raw_Volume": vol_ratio,
                "Raw_Distance": pct_from_high
            }
        except Exception:
            return None

    @st.cache_data(ttl=600)
    def run_clean_scan(tf_interval, tf_period, label_str):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(analyze_nse_stock, sym, tf_interval, tf_period, label_str): sym for sym in NIFTY_UNIVERSE}
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        return [r for r in results if r is not None]

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn2:
        if st.button("🔄 Force Re-Scan NSE", type="primary"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner(f"Analyzing all stocks on {selected_tf.split('—')[0].strip()}..."):
        all_results = run_clean_scan(interval, period, bar_label)

    stale_count = sum(1 for r in all_results if "Stale" in str(r.get("Fresh", "")))
    if stale_count > 0:
        st.warning(f"⚠️ {stale_count} of {len(all_results)} stocks returned stale data (older than expected for this timeframe).")

    st.subheader(f"🎯 1. Active Breakouts ({bar_label} High + Volume Surge)")
    breakouts = [r for r in all_results if r.get("Is_Breakout")]
    if breakouts:
        cols_bo = ["Signal", "Stock", "Price", "Breakout High", "Volume Surge", "Score", "Stop Loss", "Target", "RSI", "Fresh"]
        df_bo = pd.DataFrame(breakouts)[[c for c in cols_bo if c in pd.DataFrame(breakouts).columns]]
        st.dataframe(df_bo, use_container_width=True)
    else:
        st.info(f"ℹ️ No stocks currently meeting full breakout rules on {selected_tf.split('—')[0].strip()}.")

    st.divider()

    st.subheader("⚡ 2. Top Volume Movers Today")
    vol_movers = sorted([r for r in all_results if r.get("Is_High_Volume")], key=lambda x: x.get("Raw_Volume", 0), reverse=True)
    if vol_movers:
        cols_vol = ["Signal", "Stock", "Price", "Volume Surge", "Distance to High", "RSI", "Score", "Fresh"]
        df_vol = pd.DataFrame(vol_movers)[[c for c in cols_vol if c in pd.DataFrame(vol_movers).columns]]
        st.dataframe(df_vol, use_container_width=True)

    st.divider()

    st.subheader(f"🔭 3. Near-Breakout Radar (Within 3% of {bar_label} High)")
    near_bo = sorted([r for r in all_results if r.get("Is_Near_Breakout")], key=lambda x: x.get("Raw_Distance", 0), reverse=True)
    if near_bo:
        cols_near = ["Signal", "Stock", "Price", "Breakout High", "Distance to High", "Volume Surge", "RSI", "Score", "Fresh"]
        df_near = pd.DataFrame(near_bo)[[c for c in cols_near if c in pd.DataFrame(near_bo).columns]]
        st.dataframe(df_near, use_container_width=True)

    st.divider()

    st.subheader("🔴 4. Trend Exit Alerts (21 EMA < 50 EMA)")
    sells = [r for r in all_results if r.get("Is_Sell")]
    if sells:
        cols_sell = ["Signal", "Stock", "Price", "RSI", "Fresh"]
        df_sells = pd.DataFrame(sells)[[c for c in cols_sell if c in pd.DataFrame(sells).columns]]
        st.dataframe(df_sells, use_container_width=True)

# app.py complete
