import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime
import pytz

st.set_page_config(page_title="Professional Trading Terminal", layout="wide", page_icon="⚡")
now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))

tab1, tab2, tab3 = st.tabs([
    "🥇 Gold (XAU/USD) Dashboard", 
    "🇮🇳 NSE Stocks Scanner",
    "🎯 Intraday Option Momentum & Volatility Extension System"
])

# ========================================================
# 1. CORE UTILITIES & INDICATORS
# ========================================================
def clean_cols(df):
    if df is None or df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        l0 = [str(c).lower() for c in df.columns.get_level_values(0)]
        df.columns = df.columns.get_level_values(0 if ('close' in l0 or 'open' in l0) else 1)
    df.columns = [str(c).lower() for c in df.columns]
    return df

def calc_atr_emas(df):
    if df is None or df.empty or len(df) < 14: return df
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    return df

def plot_candles(df, title, hlines=None, emas=None, buy_pts=None, sell_pts=None, height=520):
    fig = go.Figure(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
    if emas:
        for c, clr, w in emas:
            if c in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df[c], line=dict(color=clr, width=w), name=c.upper()))
    if hlines:
        for val, clr, lbl, dsh in hlines:
            if val is not None and not np.isnan(val):
                fig.add_hline(y=val, line=dict(color=clr, width=1.5, dash=dsh), annotation_text=f"{lbl}: {val:,.2f}")
    if buy_pts is not None and not buy_pts.empty:
        fig.add_trace(go.Scatter(x=buy_pts.index, y=buy_pts.values, mode='markers', marker=dict(symbol='triangle-up', size=11, color='#00FF66'), name="BUY"))
    if sell_pts is not None and not sell_pts.empty:
        fig.add_trace(go.Scatter(x=sell_pts.index, y=sell_pts.values, mode='markers', marker=dict(symbol='triangle-down', size=11, color='#FF4D4D'), name="SELL"))
    fig.update_layout(title=title, height=height, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=40, b=20, l=20, r=20))
    return fig

# ========================================================
# TAB 1: GOLD (XAU/USD) DASHBOARD
# ========================================================
with tab1:
    g_mode = st.radio("🎯 Choose Gold Strategy:", ("Strategy 2: VWAP + 9 EMA + High/Low Levels", "Strategy 1: 125-Bar Breakout"), horizontal=True)

    if "Strategy 2" in g_mode:
        st.subheader("⚡ Gold Strategy 2: VWAP + 9 EMA + Key High/Low Levels")
        cg1, cg2 = st.columns([3, 1])
        with cg1:
            g2_tf = st.radio("⏱️ Gold Timeframe:", ("15-Minute (15M)", "30-Minute (30M)", "1-Hour (1H)", "Daily (1D)", "Monthly (1M)"), horizontal=True, index=1)
        with cg2:
            st.write("")
            if st.button("🔄 Refresh Strategy 2", type="primary"):
                st.cache_data.clear(); st.rerun()

        g_cfg = {
            "15-Minute (15M)": ("15m", "30d", "day", "PDH", "PDL", 150),
            "30-Minute (30M)": ("30m", "30d", "day", "PDH", "PDL", 150),
            "1-Hour (1H)": ("1h", "60d", "day", "PDH", "PDL", 150),
            "Daily (1D)": ("1d", "2y", "month", "PWH", "PWL", 120),
            "Monthly (1M)": ("1mo", "10y", "year", "PYH", "PYL", 48)
        }[g2_tf]

        @st.cache_data(ttl=120)
        def get_gold_vwap(iv, pr, r_mode):
            df = clean_cols(yf.download("GC=F", period=pr, interval=iv, progress=False))
            if df is None or len(df) < 15: return None, None, None
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            grp = df.index.date if r_mode == "day" else (df.index.to_period('M') if r_mode == "month" else df.index.to_period('Y'))
            tp = (df['high'] + df['low'] + df['close']) / 3.0
            df['vwap'] = (tp * df['volume']).groupby(grp).cumsum() / (df['volume'].groupby(grp).cumsum() + 1e-9)
            tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
            df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            s_keys = df.groupby(grp).size().index
            ref_df = df[grp == s_keys[-2]] if len(s_keys) >= 2 else df
            return df.dropna(subset=['ema_9', 'vwap', 'atr']), float(ref_df['high'].max()), float(ref_df['low'].min())

        gdf2, ref_h, ref_l = get_gold_vwap(g_cfg[0], g_cfg[1], g_cfg[2])
        if gdf2 is not None and len(gdf2) >= 2:
            r = gdf2.iloc[-1]
            p, v, e9, a = round(float(r['close']), 2), round(float(r['vwap']), 2), round(float(r['ema_9']), 2), round(float(r['atr']), 2)
            is_b, is_s = (p > v and e9 > v), (p < v and e9 < v)
            sig = "🟢 BUY (LONG)" if is_b else ("🔴 SELL (SHORT)" if is_s else "⚪ NEUTRAL")
            sl = f"${min(v - 1.0, p - 1.5 * a):,.2f}" if is_b else (f"${max(v + 1.0, p + 1.5 * a):,.2f}" if is_s else "-")
            tgt = f"${p + 3.0 * a:,.2f}" if is_b else (f"${p - 3.0 * a:,.2f}" if is_s else "-")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Live Gold", f"${p:,.2f}"); m2.metric(f"Signal ({g2_tf.split()[0]})", sig); m3.metric("Stop Loss", sl); m4.metric("Target (1:2)", tgt)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VWAP Line", f"${v:,.2f}"); c2.metric("9 EMA", f"${e9:,.2f}"); c3.metric(f"{g_cfg[3]} (High)", f"${ref_h:,.2f}"); c4.metric(f"{g_cfg[4]} (Low)", f"${ref_l:,.2f}")

            plot_df = gdf2.iloc[-g_cfg[5]:]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name="Gold"), 1, 1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['vwap'], line=dict(color='orange', width=2), name="VWAP"), 1, 1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_9'], line=dict(color='cyan', width=1.5), name="9 EMA"), 1, 1)
            fig.add_hline(y=ref_h, line=dict(color='#00FF66', width=1.5, dash='dash'), annotation_text=g_cfg[3], row=1, col=1)
            fig.add_hline(y=ref_l, line=dict(color='#FF4D4D', width=1.5, dash='dash'), annotation_text=g_cfg[4], row=1, col=1)
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], name="Volume"), 2, 1)
            fig.update_layout(height=580, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Could not load Gold candles for timeframe: {g2_tf}")

    else:
        st.subheader("🏆 Gold Strategy 1: 125-Bar Breakout")
        tf_choice = st.radio("Timeframe:", ("Daily (1D)", "4-Hour (4H)", "1-Hour (1H)", "15-Minute (15M)"), horizontal=True)
        iv, pr = {"Daily (1D)": ("1d", "2y"), "4-Hour (4H)": ("4h", "120d"), "1-Hour (1H)": ("1h", "60d"), "15-Minute (15M)": ("15m", "30d")}[tf_choice]

        @st.cache_data(ttl=120)
        def get_gold_bo(interval_val, period_val):
            raw = clean_cols(yf.download("GC=F", period=period_val, interval=("1h" if interval_val=="4h" else interval_val), progress=False))
            if raw is None or raw.empty: return None
            df = raw.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna() if interval_val=="4h" else raw
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            df['high_125'] = df['high'].shift(1).rolling(125).max()
            df['vol_sma_125'] = df['volume'].rolling(125).mean()
            tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
            df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            return df.dropna(subset=['high_125', 'vol_sma_125', 'atr'])

        gdf1 = get_gold_bo(iv, pr)
        if gdf1 is not None and len(gdf1) >= 2:
            r = gdf1.iloc[-1]
            p, h125, a = round(float(r['close']), 2), round(float(r['high_125']), 2), round(float(r['atr']), 2)
            vr = round(float(r['volume']) / (float(r['vol_sma_125']) + 1e-9), 2)
            is_b, is_s = (p >= h125 and vr >= 1.0), (r['ema_21'] < r['ema_50'])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Live Gold", f"${p:,.2f}")
            m2.metric("Signal", "🟢 BUY (BREAKOUT)" if is_b else ("🔴 SELL (SHORT)" if is_s else "⚪ NEUTRAL"))
            m3.metric("Stop Loss", f"${p - 2*a:,.2f}" if is_b else (f"${p + 2*a:,.2f}" if is_s else "-"))
            m4.metric("Target", f"${p + 4*a:,.2f}" if is_b else (f"${p - 4*a:,.2f}" if is_s else "-"))
            st.plotly_chart(plot_candles(gdf1.iloc[-120:], f"Gold {tf_choice}", [(h125, 'orange', '125 High', 'dash')], [('ema_21','blue',1.5), ('ema_50','red',1.5)]), use_container_width=True)

# ========================================================
# TAB 2: NSE STOCKS SCANNER
# ========================================================
with tab2:
    st.title("🇮🇳 NSE Institutional Volume & Breakout Tracker")
    sel_tf = st.radio("⏱️ Timeframe:", ("Daily (1D)", "1-Hour (1H)"), horizontal=True)
    is_d = "Daily" in sel_tf
    s_iv, s_pr = ("1d", "2y") if is_d else ("1h", "60d")

    NIFTY_UNIVERSE = [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
        "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
        "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC", "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
        "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
        "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
        "ABB", "AMBUJACEM", "BANKBARODA", "BOSCHLTD", "CANBK", "CHOLAFIN", "DABUR", "DLF", "GAIL", "GODREJCP",
        "HAVELLS", "HAL", "ICICIGI", "IOC", "IRCTC", "IRFC", "JINDALSTEL", "JIOFIN", "MOTHERSON", "PIDILITIND",
        "PNB", "PFC", "RECLTD", "SIEMENS", "TORNTPHARM", "TVSMOTOR", "VEDL", "ZOMATO", "ZYDUSLIFE", "POLYCAB"
    ]

    def scan_one(sym):
        try:
            df = clean_cols(yf.download(f"{sym}.NS", period=s_pr, interval=s_iv, progress=False))
            if df is None or len(df) < 130: return None
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            df['high_125'] = df['high'].shift(1).rolling(125).max()
            df['vol_sma_125'] = df['volume'].rolling(125).mean()
            d = df['close'].diff()
            ag = d.where(d > 0, 0.0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            al = (-d.where(d < 0, 0.0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            df['rsi'] = 100.0 - (100.0 / (1.0 + (ag / (al + 1e-9))))
            tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
            df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            df['atr_sma_50'] = df['atr'].rolling(50).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

            r, prev = df.iloc[-1], df.iloc[-2]
            p, h, a = round(float(r['close']), 2), round(float(r['high_125']), 2), round(float(r['atr']), 2)
            vr = round(float(r['volume']) / (float(r['vol_sma_125']) + 1e-9), 2)
            dist = round(((p - h) / h) * 100, 2)
            is_bo = (p >= h) and (vr >= 1.0)
            is_sell = (r['ema_21'] < r['ema_50']) and (prev['ema_21'] >= prev['ema_50'])
            is_nr = (-3.0 <= dist < 0.0) and (vr >= 0.8)
            sc = sum([int(r['rsi'] < 70), int(r['atr'] > r['atr_sma_50']), int(r['close'] > r['sma_50']), int(r['ema_21'] > r['ema_50'])])
            return {
                "Signal": "🟢 BUY" if is_bo else ("🔴 SELL" if is_sell else ("🟡 NEAR" if is_nr else "⚪ HOLD")),
                "Stock": sym, "Price": f"₹{p:,.2f}", "Breakout High": f"₹{h:,.2f}", "Distance": f"{dist:+0.2f}%",
                "Vol Surge": f"{vr}x", "RSI": round(float(r['rsi']), 1), "Score": f"{sc + (1 if is_bo else 0)}/6",
                "Stop Loss": f"₹{round(p - 2*a, 2):,.2f}", "Target": f"₹{round(p + 4*a, 2):,.2f}",
                "is_bo": is_bo, "is_vol": vr >= 1.2, "is_nr": is_nr, "is_sell": is_sell, "raw_vol": vr, "raw_dist": dist
            }
        except Exception: return None

    @st.cache_data(ttl=600)
    def run_all_scans():
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            return [r for r in ex.map(scan_one, NIFTY_UNIVERSE) if r is not None]

    if st.button("🔄 Force Re-Scan NSE", type="primary"):
        st.cache_data.clear(); st.rerun()

    with st.spinner("Screening 80 NSE assets..."):
        res = run_all_scans()

    st.subheader("🎯 1. Active Breakouts")
    bo_l = [r for r in res if r['is_bo']]
    st.dataframe(pd.DataFrame(bo_l)[["Signal", "Stock", "Price", "Breakout High", "Vol Surge", "Score", "Stop Loss", "Target", "RSI"]] if bo_l else pd.DataFrame(), use_container_width=True)

    st.subheader("⚡ 2. Top Volume Movers")
    vo_l = sorted([r for r in res if r['is_vol']], key=lambda x: x['raw_vol'], reverse=True)
    st.dataframe(pd.DataFrame(vo_l)[["Signal", "Stock", "Price", "Vol Surge", "Distance", "RSI", "Score"]] if vo_l else pd.DataFrame(), use_container_width=True)

    st.subheader("🔭 3. Near-Breakout Radar")
    nr_l = sorted([r for r in res if r['is_nr']], key=lambda x: x['raw_dist'], reverse=True)
    st.dataframe(pd.DataFrame(nr_l)[["Signal", "Stock", "Price", "Breakout High", "Distance", "Vol Surge", "RSI", "Score"]] if nr_l else pd.DataFrame(), use_container_width=True)

    st.subheader("🔴 4. Trend Exit Alerts")
    ex_l = [r for r in res if r['is_sell']]
    st.dataframe(pd.DataFrame(ex_l)[["Signal", "Stock", "Price", "RSI"]] if ex_l else pd.DataFrame(), use_container_width=True)

# ========================================================
# TAB 3: INTRADAY OPTION MOMENTUM & MULTI-TIMEFRAME ENGINE
# ========================================================
with tab3:
    st.header("🎯 Intraday Option Momentum & Volatility Extension System")
    st.caption("Multi-Timeframe Engine: Independent Signals, Fail Levels & 5-Tier Target Ladders on 15M, 1H, Daily & Monthly")

    PRESETS = {
        "NIFTY 50 (NSE)": "^NSEI", "BANK NIFTY (NSE)": "^NSEBANK", "FIN NIFTY (NSE)": "NIFTY_FIN_SERVICE.NS",
        "MIDCP NIFTY (NSE)": "NIFTY_MID_SELECT.NS", "SENSEX (BSE)": "^BSESN", "BANKEX (BSE)": "BSE-BANK.BO",
        "COMEX Gold": "GC=F", "Custom": "CUSTOM"
    }

    cs1, cs2, cs3 = st.columns([1.6, 1.4, 1.0])
    with cs1: chosen = st.selectbox("Select Asset / Index:", list(PRESETS.keys()))
    with cs2:
        u_sym = st.text_input("Ticker Symbol:", value="RELIANCE.NS" if chosen == "Custom" else PRESETS[chosen], disabled=(chosen != "Custom")).strip().upper()
    with cs3:
        st.write(""); st.write("")
        if st.button("🔄 Recalculate Multi-TF", type="primary"):
            st.cache_data.clear(); st.rerun()

    @st.cache_data(ttl=60)
    def fetch_m_tf(sym):
        d15 = calc_atr_emas(clean_cols(yf.download(sym, period="30d", interval="15m", progress=False)))
        d1h = calc_atr_emas(clean_cols(yf.download(sym, period="60d", interval="1h", progress=False)))
        d1d = calc_atr_emas(clean_cols(yf.download(sym, period="2y", interval="1d", progress=False)))
        d1m = calc_atr_emas(clean_cols(yf.download(sym, period="10y", interval="1mo", progress=False)))
        return d15, d1h, d1d, d1m

    with st.spinner(f"Loading 15M, 1H, Daily, and Monthly feeds for {u_sym}..."):
        d15, d1h, d1d, d1m = fetch_m_tf(u_sym)

    def render_opt_tf(df, tf_name, extra_h=None, extra_e=None, bars=90):
        if df is None or df.empty or len(df) < 15:
            st.warning(f"Insufficient data for {tf_name}."); return

        r = df.iloc[-1]
        c_p, c_a = float(r['close']), (float(r['atr']) if not np.isnan(r['atr']) else float(r['close']) * 0.01)
        c_e9, c_e21 = float(r['ema_9']), float(r['ema_21'])

        is_bull = (c_p >= c_e9) and (c_e9 >= c_e21)
        is_bear = (c_p < c_e9) and (c_e9 < c_e21)
        step = round(c_a * 1.5, 2)
        h_step = round(step / 2.0, 2)

        bull_s = (df['close'] >= df['ema_9']) & (df['ema_9'] >= df['ema_21'])
        bear_s = (df['close'] < df['ema_9']) & (df['ema_9'] < df['ema_21'])
        b_pts = df[bull_s & (~bull_s.shift(1).fillna(False))]['low'] * 0.999
        s_pts = df[bear_s & (~bear_s.shift(1).fillna(False))]['high'] * 1.001

        if is_bull:
            badge, act, title, bg, border = "🟢 BUY (LONG / CALL)", "BUY", f"CONTINUE VIEW UP — 🟢 BUY SETUP ACTIVE ({tf_name})", "#0d3b1e", "#00FF66"
            f_lvl, f_str = round(c_p - step, 2), f"{round(c_p - step, 2):,.2f} (-{step:,.2f} pts)"
            r1 = round(c_p + step, 2)
            ladder = [(f"Target {i+1}", round(r1 + i * h_step, 2)) for i in range(5)]
        elif is_bear:
            badge, act, title, bg, border = "🔴 SELL (SHORT / PUT)", "SELL", f"CONTINUE VIEW DOWN — 🔴 SELL SETUP ACTIVE ({tf_name})", "#421616", "#FF4D4D"
            f_lvl, f_str = round(c_p + step, 2), f"{round(c_p + step, 2):,.2f} (+{step:,.2f} pts)"
            r1 = round(c_p - step, 2)
            ladder = [(f"Target {i+1}", round(r1 - i * h_step, 2)) for i in range(5)]
        else:
            badge, act, title, bg, border = "⚪ NEUTRAL / WAIT", "NEUTRAL", f"CONSOLIDATION — ⚪ NO CLEAR SIGNAL ({tf_name})", "#222222", "#888888"
            f_lvl, f_str = round(c_p - step, 2), f"{round(c_p - step, 2):,.2f} ({step:,.2f} pts)"
            r1 = round(c_p + step, 2)
            ladder = [(f"Range {i+1}", round(r1 + i * h_step, 2)) for i in range(5)]

        # KPI Row
        k1, k2, k3, k4, k5 = st.columns([1.5, 1.2, 1.3, 1.2, 1.2])
        k1.metric("Signal", badge); k2.metric("Current Price", f"{c_p:,.2f}"); k3.metric("Stop Loss (Fail LVL)", f"{f_lvl:,.2f}")
        k4.metric("Target 1", f"{ladder[0][1]:,.2f}"); k5.metric("Target 3", f"{ladder[2][1]:,.2f}")

        # HUD Badge Card
        badges_html = "".join([f"<div style='background:rgba(255,255,255,0.08);padding:8px 12px;border-radius:6px;min-width:130px;margin:3px;display:inline-block;'><div style='color:#B0B0B0;font-size:12px;'>{n}</div><div style='color:#FFF;font-weight:bold;font-size:16px;'>{l:,.2f}</div></div>" for n, l in ladder])
        st.markdown(f"<div style='background-color:{bg};border:2px solid {border};border-radius:10px;padding:14px;margin-bottom:12px;'><h3 style='color:#FFF;margin:0;'>{title}</h3><div style='color:#E0E0E0;font-size:13px;margin:3px 0 8px 0;'>Current Close: <b>{c_p:,.2f}</b> | Mode: <b>{act}</b></div><div style='background:rgba(0,0,0,0.35);padding:6px 12px;border-radius:6px;margin-bottom:8px;'><span style='color:#FFAA00;font-weight:bold;'>Move View Fail LVL: </span><span style='color:#FFF;font-weight:bold;font-size:16px;'>{f_str}</span></div><div style='display:flex;flex-wrap:wrap;gap:4px;'>{badges_html}</div></div>", unsafe_allow_html=True)

        h_lines = [(f_lvl, '#FFAA00', 'Fail LVL', 'dot')] + [(l, '#00FF66' if is_bull else '#FF4D4D', n, 'dash') for n, l in ladder]
        if extra_h: h_lines.extend(extra_h)
        emas = [('ema_9', 'cyan', 1.5), ('ema_21', 'magenta', 1.5)]
        if extra_e: emas.extend(extra_e)

        st.plotly_chart(plot_candles(df.iloc[-bars:], f"{tf_name} Targets ({u_sym})", h_lines, emas, b_pts, s_pts, height=540), use_container_width=True)

    ct1, ct2, ct3, ct4 = st.tabs(["⚡ 15-Minute", "⏰ 1-Hour", "📅 Daily", "🏛️ Monthly"])
    with ct1: render_opt_tf(d15, "15-Minute Execution", bars=90)
    with ct2: render_opt_tf(d1h, "1-Hour Swing", bars=90)
    with ct3:
        ex_d = []
        if d1d is not None and len(d1d) >= 2:
            prev_d = d1d.iloc[-2]
            ex_d = [(float(prev_d['high']), '#00FF66', 'PDH', 'dash'), (float(prev_d['low']), '#FF4D4D', 'PDL', 'dash')]
        render_opt_tf(d1d, "Daily Positional", extra_h=ex_d, extra_e=[('ema_50', 'orange', 1.5)], bars=120)
    with ct4:
        ex_m = []
        if d1m is not None and len(d1m) >= 2:
            prev_m = d1m.iloc[-2]
            ex_m = [(float(prev_m['high']), '#00FF66', 'PMH', 'dash'), (float(prev_m['low']), '#FF4D4D', 'PML', 'dash')]
        render_opt_tf(d1m, "Monthly Macro", extra_h=ex_m, bars=48)

# === END OF APY.PY ===
