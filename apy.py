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

# Top Navigation Tabs
main_tab1, main_tab2 = st.tabs(["🇮🇳 NSE Stocks Scanner", "🥇 Gold (XAU/USD) Dashboard"])

# ========================================================
# CORE INDICATOR ENGINE (Rules A through K)
# ========================================================
def compute_indicators(df):
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

    return df.dropna(subset=['high_125', 'vol_sma_125', 'rsi', 'atr'])

# ========================================================
# TAB 1: COMPLETE NSE STOCKS SCANNER
# ========================================================
with main_tab1:
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

    def analyze_nse_stock(symbol, tf_interval, tf_period):
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

            is_sell = (row['ema_21'] < row['ema_50']) and (prev['ema_21'] >= prev['ema_50'])
            confirm_score = sum([cond_c, cond_d, cond_e, cond_f])

            return {
                "Stock": symbol,
                "Price": f"₹{close:,.2f}",
                "Raw_Price": close,
                f"{bar_label} High": f"₹{high_125:,.2f}",
                "Distance to High": f"{pct_from_high:+0.2f}%",
                "Raw_Distance": pct_from_high,
                "Volume Surge": f"{vol_ratio}x Avg",
                "Raw_Volume": vol_ratio,
                "RSI": round(row['rsi'], 1),
                "Score": f"{confirm_score + (1 if cond_a else 0) + (1 if cond_b else 0)}/6",
                "Is_Breakout": cond_a and cond_b,
                "Is_Near_Breakout": (-3.0 <= pct_from_high < 0.0) and (vol_ratio >= 0.8),
                "Is_High_Volume": vol_ratio >= 1.2,
                "Is_Sell": is_sell,
                "Stop_Loss": f"₹{round(close - (2 * atr_val), 2):,.2f}",
                "Target": f"₹{round(close + (4 * atr_val), 2):,.2f}"
            }
        except:
            return None

    @st.cache_data(ttl=600)
    def run_nse_scan(tf_interval, tf_period):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(analyze_nse_stock, sym, tf_interval, tf_period): sym for sym in NIFTY_UNIVERSE}
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        return [r for r in results if r is not None]

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn2:
        if st.button("🔄 Force Re-Scan NSE", type="primary"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner(f"Analyzing all 80 stocks on {selected_tf.split('—')[0].strip()}..."):
        all_results = run_nse_scan(interval, period)

    # Section 1: Active Breakouts
    st.subheader(f"🎯 1. Active Breakouts ({bar_label} High + Volume Surge)")
    breakouts = [r for r in all_results if r["Is_Breakout"]]
    if breakouts:
        df_bo = pd.DataFrame(breakouts)[["Stock", "Price", f"{bar_label} High", "Volume Surge", "Score", "Stop_Loss", "Target", "RSI"]]
        st.dataframe(df_bo, use_container_width=True)
    else:
        st.info(f"ℹ️ No stocks currently at exact 0% breakout with volume surge on {selected_tf.split('—')[0].strip()}. Check Section 2 below for active volume!")

    st.divider()

    # Section 2: Top Volume Movers
    st.subheader("⚡ 2. Top Volume Movers Today (Institutional Buying Radar)")
    st.write("Stocks where current volume is substantially higher than the 125-bar average, showing active money inflow:")
    vol_movers = sorted([r for r in all_results if r["Is_High_Volume"]], key=lambda x: x["Raw_Volume"], reverse=True)
    if vol_movers:
        df_vol = pd.DataFrame(vol_movers)[["Stock", "Price", "Volume Surge", "Distance to High", "RSI", "Score"]]
        st.dataframe(df_vol, use_container_width=True)
    else:
        st.info("ℹ️ No unusual volume surges (>1.2x average) detected right now.")

    st.divider()

    # Section 3: Near-Breakout Radar
    st.subheader(f"🔭 3. Near-Breakout Radar (Within 3% of {bar_label} High)")
    st.write("Stocks setting up right below the breakout ceiling with strong volume — catch them before they cross:")
    near_bo = sorted([r for r in all_results if r["Is_Near_Breakout"]], key=lambda x: x["Raw_Distance"], reverse=True)
    if near_bo:
        df_near = pd.DataFrame(near_bo)[["Stock", "Price", f"{bar_label} High", "Distance to High", "Volume Surge", "RSI", "Score"]]
        st.dataframe(df_near, use_container_width=True)
    else:
        st.info(f"ℹ️ No stocks currently within 3% of their {bar_label} high.")

    st.divider()

    # Section 4: Exit Alerts
    st.subheader("🔴 4. Trend Exit Alerts (21 EMA < 50 EMA)")
    sells = [r for r in all_results if r["Is_Sell"]]
    if sells:
        df_sells = pd.DataFrame(sells)[["Stock", "Price", "RSI"]]
        st.dataframe(df_sells, use_container_width=True)
    else:
        st.info("ℹ️ No fresh sell crossovers on this timeframe.")

    st.divider()

    # Section 5: Interactive Chart
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

# ========================================================
# TAB 2: GOLD (XAU/USD) DASHBOARD
# ========================================================
with main_tab2:
    st.title("🥇 Gold (XAU/USD) Real-Time Strategy Dashboard")
    st.caption("Tracking Spot / COMEX Futures (GC=F) with real volume, 125-bar breakout levels, and ATR risk rules.")

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
        "Daily (1D) — Macro Swing": ("1d", "2y", "125-Day"),
        "4-Hour (4H) — Swing": ("1h", "120d", "125-Bar"),
        "1-Hour (1H) — Intraday / Short-Term": ("1h", "60d", "125-Hour"),
        "15-Minute (15M) — Scalp": ("15m", "30d", "125-Bar (15M)")
    }
    g_interval, g_period, g_bar_label = tf_map[gold_tf]

    @st.cache_data(ttl=120)
    def fetch_gold(interval_val, period_val):
        df = yf.download("GC=F", period=period_val, interval=interval_val, progress=False)
        return compute_indicators(df)

    with st.spinner("Fetching live Gold market data..."):
        gold_df = fetch_gold(g_interval, g_period)

    if gold_df is None or len(gold_df) < 2:
        st.error("Could not fetch sufficient Gold market candles. Try another timeframe or refresh.")
    else:
        row = gold_df.iloc[-1]
        prev = gold_df.iloc[-2]

        close = round(row['close'], 2)
        high_125 = round(row['high_125'], 2)
        atr_val = round(row['atr'], 2)
        vol_ratio = round(row['volume'] / (row['vol_sma_125'] + 1e-9), 2)
        pct_diff = round(((close - high_125) / high_125) * 100, 2)

        c_a = close >= high_125
        c_b = vol_ratio >= 1.0
        c_c = row['rsi'] < 70
        c_d = row['atr'] > row['atr_sma_50']
        c_e = row['close'] > row['sma_50']
        c_f = row['ema_21'] > row['ema_50']
        sell_crossover = (row['ema_21'] < row['ema_50']) and (prev['ema_21'] >= prev['ema_50'])

        score = sum([c_a, c_b, c_c, c_d, c_e, c_f])

        if c_a and c_b:
            sig = "🟢 STRONG BREAKOUT BUY"
            sig_color = "#00C853"
        elif score >= 4:
            sig = "🟡 BULLISH TREND (Waiting for Breakout)"
            sig_color = "#FFAB00"
        elif sell_crossover:
            sig = "🔴 SELL / EXIT (EMA Breakdown)"
            sig_color = "#D50000"
        else:
            sig = "⚪ NEUTRAL / CONSOLIDATION"
            sig_color = "#888888"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Gold Price (USD)", f"${close:,.2f}")
        m2.markdown(f"**Signal**  \n<h3 style='color:{sig_color};margin:0;'>{sig}</h3>", unsafe_allow_html=True)
        m3.metric(f"Stop-Loss (-2 ATR)", f"${round(close - (2 * atr_val), 2):,.2f}")
        m4.metric(f"Target (+4 ATR, 1:2 RR)", f"${round(close + (4 * atr_val), 2):,.2f}")

        st.divider()

        st.subheader(f"Gold Checklist Score: {score}/6 Conditions Met")
        k1, k2, k3 = st.columns(3)
        k4, k5, k6 = st.columns(3)

        k1.success(f"✅ A: {g_bar_label} Breakout" if c_a else f"❌ A: Below {g_bar_label} High (${high_125:,.2f}) [{pct_diff:+0.2f}%]")
        k2.success(f"✅ B: Volume Surge ({vol_ratio}x Avg)" if c_b else f"❌ B: Low Volume ({vol_ratio}x Avg)")
        k3.success(f"✅ C: Safe RSI ({row['rsi']:.1f} < 70)" if c_c else f"❌ C: Overbought RSI ({row['rsi']:.1f})")
        k4.success("✅ D: Expanding Volatility" if c_d else "❌ D: Low Volatility")
        k5.success("✅ E: Price > 50 SMA" if c_e else "❌ E: Price Below 50 SMA")
        k6.success("✅ F: 21 EMA > 50 EMA" if c_f else "❌ F: 21 EMA < 50 EMA")

        st.divider()

        st.subheader(f"Gold (XAU/USD) Technical Chart ({g_interval.upper()})")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])

        fig.add_trace(go.Candlestick(x=gold_df.index, open=gold_df['open'], high=gold_df['high'], low=gold_df['low'], close=gold_df['close'], name="Gold Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['high_125'], line=dict(color='orange', dash='dash'), name=f"{g_bar_label} High"), row=1, col=1)
        fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['ema_21'], line=dict(color='blue', width=1.5), name="21 EMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['ema_50'], line=dict(color='red', width=1.5), name="50 EMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['sma_50'], line=dict(color='yellow', width=1.5), name="50 SMA"), row=1, col=1)

        v_colors = ['green' if c >= o else 'red' for c, o in zip(gold_df['close'], gold_df['open'])]
        fig.add_trace(go.Bar(x=gold_df.index, y=gold_df['volume'], marker_color=v_colors, name="COMEX Volume"), row=2, col=1)
        fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['vol_sma_125'], line=dict(color='purple'), name="125-Avg Volume"), row=2, col=1)

        fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['rsi'], line=dict(color='magenta', width=1.5), name="RSI (14)"), row=3, col=1)
        fig.add_hline(y=70, line=dict(color='red', dash='dot'), row=3, col=1)
        fig.add_hline(y=30, line=dict(color='green', dash='dot'), row=3, col=1)

        fig.update_layout(height=750, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
