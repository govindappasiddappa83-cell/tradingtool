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

st.set_page_config(page_title="Professional Trading Terminal", layout="wide", page_icon="⚡")
IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

tab1, tab2, tab3 = st.tabs([
    "🥇 Gold (XAU/USD) Dashboard", 
    "🇮🇳 NSE Stocks Scanner",
    "🎯 Intraday Option Momentum & Volatility Extension System"
])

# ========================================================
# UTILITIES & REUSABLE INDICATORS
# ========================================================
def clean_cols(df):
    if df is None or df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        l0 = [str(c).lower() for c in df.columns.get_level_values(0)]
        df.columns = df.columns.get_level_values(0 if ('close' in l0 or 'open' in l0) else 1)
    df.columns = [str(c).lower() for c in df.columns]
    return df

def add_indicators(df):
    if df is None or df.empty or len(df) < 130: return None
    df = clean_cols(df)
    req = ['open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in req): return None
    df = df[req].copy()
    df['high_125'] = df['high'].shift(1).rolling(125).max()
    df['low_125'] = df['low'].shift(1).rolling(125).min()
    df['vol_sma_125'] = df['volume'].rolling(125).mean()

    # Wilder RMA RSI
    d = df['close'].diff()
    ag = d.where(d > 0, 0.0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = (-d.where(d < 0, 0.0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['rsi'] = 100.0 - (100.0 / (1.0 + (ag / (al + 1e-9))))

    # ATR & MAs
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['atr_sma_50'] = df['atr'].rolling(50).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(min(200, len(df)-1)).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    return df.dropna(subset=['high_125', 'vol_sma_125', 'rsi', 'atr'])

def plot_candles(df, title, hlines=None, emas=None, height=520):
    fig = go.Figure(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"))
    if emas:
        for c, clr, w in emas:
            if c in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df[c], line=dict(color=clr, width=w), name=c.upper()))
    if hlines:
        for val, clr, lbl, dsh in hlines:
            if val is not None and not np.isnan(val):
                fig.add_hline(y=val, line=dict(color=clr, width=1.5, dash=dsh), annotation_text=f"{lbl}: {val:,.2f}")
    fig.update_layout(title=title, height=height, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=40, b=20, l=20, r=20))
    return fig

# ========================================================
# TAB 1: GOLD TERMINAL
# ========================================================
with tab1:
    g_mode = st.radio("🎯 Choose Gold Strategy:", ("Strategy 2: 30M VWAP + 9 EMA + PDH/PDL", "Strategy 1: 125-Bar Breakout"), horizontal=True)

    if "Strategy 2" in g_mode:
        st.subheader("⚡ Gold Strategy 2: 30M Session VWAP + 9 EMA + PDH/PDL")
        if st.button("🔄 Refresh 30M Gold", type="primary"):
            st.cache_data.clear(); st.rerun()

        @st.cache_data(ttl=120)
        def get_gold_30m():
            df = clean_cols(yf.download("GC=F", period="30d", interval="30m", progress=False))
            if df is None or len(df) < 50: return None, None, None
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['date'] = df.index.date
            tp = (df['high'] + df['low'] + df['close']) / 3.0
            df['vwap'] = (tp * df['volume']).groupby(df['date']).cumsum() / (df['volume'].groupby(df['date']).cumsum() + 1e-9)
            tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
            df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            vdays = df.groupby('date').size()[lambda x: x >= 10].index
            pdh = float(df[df['date'] == vdays[-2]]['high'].max()) if len(vdays) >= 2 else float(df['high'].max())
            pdl = float(df[df['date'] == vdays[-2]]['low'].min()) if len(vdays) >= 2 else float(df['low'].min())
            return df.dropna(subset=['ema_9', 'vwap', 'atr']), pdh, pdl

        df_30m, pdh_g, pdl_g = get_gold_30m()
        if df_30m is not None and len(df_30m) >= 2:
            r = df_30m.iloc[-1]
            p, v, e9, a = round(float(r['close']), 2), round(float(r['vwap']), 2), round(float(r['ema_9']), 2), round(float(r['atr']), 2)
            is_b, is_s = (p > v and e9 > v), (p < v and e9 < v)
            sig = "🟢 BUY" if is_b else ("🔴 SELL" if is_s else "⚪ NEUTRAL")
            sl = f"${min(v - 1.0, p - 1.5 * a):,.2f}" if is_b else (f"${max(v + 1.0, p + 1.5 * a):,.2f}" if is_s else "-")
            tgt = f"${p + 3.0 * a:,.2f}" if is_b else (f"${p - 3.0 * a:,.2f}" if is_s else "-")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Live Gold", f"${p:,.2f}"); m2.metric("Signal", sig); m3.metric("Stop Loss", sl); m4.metric("Target", tgt)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VWAP", f"${v:,.2f}"); c2.metric("9 EMA", f"${e9:,.2f}"); c3.metric("PDH", f"${pdh_g:,.2f}"); c4.metric("PDL", f"${pdl_g:,.2f}")

            plot_df = df_30m.iloc[-120:]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name="Gold"), 1, 1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['vwap'], line=dict(color='orange', width=2), name="VWAP"), 1, 1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_9'], line=dict(color='cyan', width=1.5), name="9 EMA"), 1, 1)
            fig.add_hline(y=pdh_g, line=dict(color='#00FF66', width=1.5, dash='dash'), annotation_text="PDH", row=1, col=1)
            fig.add_hline(y=pdl_g, line=dict(color='#FF4D4D', width=1.5, dash='dash'), annotation_text="PDL", row=1, col=1)
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], name="Volume"), 2, 1)
            fig.update_layout(height=580, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("🏆 Gold Strategy 1: 125-Bar Breakout")
        tf_choice = st.radio("Timeframe:", ("Daily (1D)", "4-Hour (4H)", "1-Hour (1H)", "15-Minute (15M)"), horizontal=True)
        tf_dict = {"Daily (1D)": ("1d", "2y"), "4-Hour (4H)": ("4h", "120d"), "1-Hour (1H)": ("1h", "60d"), "15-Minute (15M)": ("15m", "30d")}
        iv, pr = tf_dict[tf_choice]

        @st.cache_data(ttl=120)
        def get_gold_tf(interval_val, period_val):
            if interval_val == "4h":
                raw = clean_cols(yf.download("GC=F", period=period_val, interval="1h", progress=False))
                df = raw.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna() if raw is not None else None
            else:
                df = yf.download("GC=F", period=period_val, interval=interval_val, progress=False)
            return add_indicators(df)

        gdf = get_gold_tf(iv, pr)
        if gdf is not None and len(gdf) >= 2:
            r = gdf.iloc[-1]
            p, h125, a = round(float(r['close']), 2), round(float(r['high_125']), 2), round(float(r['atr']), 2)
            vr = round(float(r['volume']) / (float(r['vol_sma_125']) + 1e-9), 2)
            is_b, is_s = (p >= h125 and vr >= 1.0), (r['ema_21'] < r['ema_50'])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Live Gold", f"${p:,.2f}")
            m2.metric("Signal", "🟢 BUY" if is_b else ("🔴 SELL" if is_s else "⚪ NEUTRAL"))
            m3.metric("Stop Loss", f"${p - 2*a:,.2f}" if is_b else (f"${p + 2*a:,.2f}" if is_s else "-"))
            m4.metric("Target", f"${p + 4*a:,.2f}" if is_b else (f"${p - 4*a:,.2f}" if is_s else "-"))
            st.plotly_chart(plot_candles(gdf.iloc[-120:], f"Gold {tf_choice}", [(h125, 'orange', '125 High', 'dash')], [('ema_21','blue',1.5), ('ema_50','red',1.5)]), use_container_width=True)

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
            df = yf.download(f"{sym}.NS", period=s_pr, interval=s_iv, progress=False)
            df = add_indicators(df)
            if df is None or len(df) < 2: return None
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
    bo_list = [r for r in res if r['is_bo']]
    st.dataframe(pd.DataFrame(bo_list)[["Signal", "Stock", "Price", "Breakout High", "Vol Surge", "Score", "Stop Loss", "Target", "RSI"]] if bo_list else pd.DataFrame(), use_container_width=True)

    st.subheader("⚡ 2. Top Volume Movers")
    vol_list = sorted([r for r in res if r['is_vol']], key=lambda x: x['raw_vol'], reverse=True)
    st.dataframe(pd.DataFrame(vol_list)[["Signal", "Stock", "Price", "Vol Surge", "Distance", "RSI", "Score"]] if vol_list else pd.DataFrame(), use_container_width=True)

    st.subheader("🔭 3. Near-Breakout Radar")
    nr_list = sorted([r for r in res if r['is_nr']], key=lambda x: x['raw_dist'], reverse=True)
    st.dataframe(pd.DataFrame(nr_list)[["Signal", "Stock", "Price", "Breakout High", "Distance", "Vol Surge", "RSI", "Score"]] if nr_list else pd.DataFrame(), use_container_width=True)

    st.subheader("🔴 4. Trend Exit Alerts")
    ex_list = [r for r in res if r['is_sell']]
    st.dataframe(pd.DataFrame(ex_list)[["Signal", "Stock", "Price", "RSI"]] if ex_list else pd.DataFrame(), use_container_width=True)

# ========================================================
# TAB 3: INTRADAY OPTION MOMENTUM & MULTI-TIMEFRAME ENGINE
# ========================================================
with tab3:
    st.header("🎯 Intraday Option Momentum & Volatility Extension System")
    st.caption("Multi-Timeframe Engine: 15M Ladder, 1H Momentum, Daily PDH/PDL, Monthly PMH/PML")

    PRESETS = {
        "NIFTY 50 (NSE)": "^NSEI", "BANK NIFTY (NSE)": "^NSEBANK", "FIN NIFTY (NSE)": "NIFTY_FIN_SERVICE.NS",
        "MIDCP NIFTY (NSE)": "NIFTY_MID_SELECT.NS", "SENSEX (BSE)": "^BSESN", "BANKEX (BSE)": "BSE-BANK.BO",
        "COMEX Gold": "GC=F", "Custom": "CUSTOM"
    }

    c_s1, c_s2, c_s3 = st.columns([1.6, 1.4, 1.0])
    with c_s1: chosen = st.selectbox("Select Asset / Index:", list(PRESETS.keys()))
    with c_s2:
        u_sym = st.text_input("Ticker Symbol:", value="RELIANCE.NS" if chosen == "Custom" else PRESETS[chosen], disabled=(chosen != "Custom")).strip().upper()
    with c_s3:
        st.write(""); st.write("")
        if st.button("🔄 Recalculate Multi-TF", type="primary"):
            st.cache_data.clear(); st.rerun()

    @st.cache_data(ttl=60)
    def fetch_m_tf(sym):
        d15 = clean_cols(yf.download(sym, period="30d", interval="15m", progress=False))
        d1h = clean_cols(yf.download(sym, period="60d", interval="1h", progress=False))
        d1d = clean_cols(yf.download(sym, period="1y", interval="1d", progress=False))
        d1m = clean_cols(yf.download(sym, period="5y", interval="1mo", progress=False))
        return d15, d1h, d1d, d1m

    with st.spinner(f"Loading 15M, 1H, Daily, and Monthly data for {u_sym}..."):
        d15, d1h, d1d, d1m = fetch_m_tf(u_sym)

    if d15 is None or d15.empty or len(d15) < 15:
        st.error(f"Could not load intraday data for '{u_sym}'. Please verify the symbol.")
    else:
        # 15M Indicators
        tr_15 = pd.concat([d15['high'] - d15['low'], (d15['high'] - d15['close'].shift(1)).abs(), (d15['low'] - d15['close'].shift(1)).abs()], axis=1).max(axis=1)
        d15['atr'] = tr_15.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        d15['ema_9'] = d15['close'].ewm(span=9, adjust=False).mean()
        d15['ema_21'] = d15['close'].ewm(span=21, adjust=False).mean()

        r15 = d15.iloc[-1]
        c_p, c_a = float(r15['close']), float(r15['atr'])
        c_e9, c_e21 = float(r15['ema_9']), float(r15['ema_21'])

        is_bull = (c_p >= c_e9) and (c_e9 >= c_e21)
        is_bear = (c_p < c_e9) and (c_e9 < c_e21)
        step = round(c_a * 1.5, 2)
        h_step = round(step / 2.0, 2)
        lbl = chosen.split('(')[0].strip()

        if is_bull:
            b_title, bg, border = f"CONTINUE VIEW UP — {lbl}", "#113b22", "#00FF66"
            f_lvl = round(c_p - step, 2)
            f_str = f"{f_lvl:,.2f} ({step:,.2f} pts)"
            r1 = round(c_p + step, 2)
            ladder = [(f"N Upper Range {i+1}", round(r1 + i * h_step, 2)) for i in range(5)]
        elif is_bear:
            b_title, bg, border = f"CONTINUE VIEW DOWN — {lbl}", "#421616", "#FF4D4D"
            f_lvl = round(c_p + step, 2)
            f_str = f"{f_lvl:,.2f} ({step:,.2f} pts)"
            r1 = round(c_p - step, 2)
            ladder = [(f"N Lower Range {i+1}", round(r1 - i * h_step, 2)) for i in range(5)]
        else:
            b_title, bg, border = f"NEUTRAL / CONSOLIDATION — {lbl}", "#222222", "#888888"
            f_lvl = round(c_p - step, 2)
            f_str = f"{f_lvl:,.2f} ({step:,.2f} pts)"
            r1 = round(c_p + step, 2)
            ladder = [(f"Range Level {i+1}", round(r1 + i * h_step, 2)) for i in range(5)]

        # Daily PDH/PDL
        pdh_v, pdl_v = None, None
        if d1d is not None and len(d1d) >= 2:
            d1d['ema_21'] = d1d['close'].ewm(span=21, adjust=False).mean()
            d1d['ema_50'] = d1d['close'].ewm(span=50, adjust=False).mean()
            prev_d = d1d.iloc[-2]
            pdh_v, pdl_v = float(prev_d['high']), float(prev_d['low'])

        # 1-Hour indicators
        if d1h is not None and len(d1h) >= 21:
            d1h['ema_9'] = d1h['close'].ewm(span=9, adjust=False).mean()
            d1h['ema_21'] = d1h['close'].ewm(span=21, adjust=False).mean()

        # Monthly PMH/PML
        pmh_v, pml_v = None, None
        if d1m is not None and len(d1m) >= 2:
            prev_m = d1m.iloc[-2]
            pmh_v, pml_v = float(prev_m['high']), float(prev_m['low'])

        # HUD Card
        time_str = now_ist.strftime("%d %b %I:%M %p (IST)")
        hud_badges = "".join([f"<div style='background:rgba(255,255,255,0.08);padding:8px 12px;border-radius:6px;min-width:140px;margin:3px;display:inline-block;'><div style='color:#B0B0B0;font-size:12px;'>{n}</div><div style='color:#FFFFFF;font-weight:bold;font-size:16px;'>{l:,.2f}</div></div>" for n, l in ladder])
        st.markdown(f"<div style='background-color:{bg};border:2px solid {border};border-radius:10px;padding:16px;margin-bottom:12px;'><h2 style='color:#FFF;margin:0;'>{b_title}</h2><div style='color:#E0E0E0;font-size:13px;margin:4px 0 10px 0;'>📅 {time_str} | Price: <b>{c_p:,.2f}</b></div><div style='background:rgba(0,0,0,0.35);padding:8px 14px;border-radius:6px;margin-bottom:10px;'><span style='color:#FFAA00;font-weight:bold;'>Move View Fail LVL: </span><span style='color:#FFF;font-weight:bold;font-size:17px;'>{f_str}</span></div><div style='display:flex;flex-wrap:wrap;gap:4px;'>{hud_badges}</div></div>", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Current Price", f"{c_p:,.2f}")
        k2.metric("Fail Level", f"{f_lvl:,.2f}")
        k3.metric("PDH (Prev Day High)", f"{pdh_v:,.2f}" if pdh_v else "N/A")
        k4.metric("PDL (Prev Day Low)", f"{pdl_v:,.2f}" if pdl_v else "N/A")
        st.divider()

        # 4 Multi-Timeframe Tabs
        ct1, ct2, ct3, ct4 = st.tabs(["⚡ 15-Minute (Ladder)", "📅 Daily (PDH/PDL)", "⏰ 1-Hour (Trend)", "🏛️ Monthly (PMH/PML)"])

        with ct1:
            h_15 = [(f_lvl, '#FFAA00', 'Fail LVL', 'dot')] + [(l, '#00FF66' if is_bull else '#FF4D4D', n, 'dash') for n, l in ladder]
            st.plotly_chart(plot_candles(d15.iloc[-90:], f"15-Minute Target Ladder ({u_sym})", h_15, [('ema_9', 'cyan', 1.5), ('ema_21', 'magenta', 1.5)], height=560), use_container_width=True)

        with ct2:
            if d1d is not None and len(d1d) >= 10:
                h_d = [(pdh_v, '#00FF66', 'PDH', 'dash'), (pdl_v, '#FF4D4D', 'PDL', 'dash')]
                st.plotly_chart(plot_candles(d1d.iloc[-120:], f"Daily Key Levels ({u_sym})", h_d, [('ema_21', 'blue', 1.5), ('ema_50', 'orange', 1.5)], height=560), use_container_width=True)
            else: st.info("Daily data not available.")

        with ct3:
            if d1h is not None and len(d1h) >= 20:
                st.plotly_chart(plot_candles(d1h.iloc[-90:], f"1-Hour Swing Momentum ({u_sym})", None, [('ema_9', 'cyan', 1.5), ('ema_21', 'magenta', 1.5)], height=560), use_container_width=True)
            else: st.info("1-Hour data not available.")

        with ct4:
            if d1m is not None and len(d1m) >= 5:
                h_m = [(pmh_v, '#00FF66', 'PMH', 'dash'), (pml_v, '#FF4D4D', 'PML', 'dash')]
                st.plotly_chart(plot_candles(d1m.iloc[-36:], f"Monthly Macro Levels ({u_sym})", h_m, height=560), use_container_width=True)
            else: st.info("Monthly data not available.")

# === END OF APY.PY ===
