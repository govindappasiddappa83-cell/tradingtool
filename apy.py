import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pytz

st.set_page_config(page_title="Professional Trading Terminal", layout="wide", page_icon="⚡")
now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))

tab1, tab2 = st.tabs([
    "🥇 Gold (XAU/USD) Dashboard", 
    "🎯 Intraday Option Momentum & Volatility Extension System"
])

# ========================================================
# 1. CORE UTILITIES & SHARED HELPERS
# ========================================================
def clean_cols(df):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        l0 = [str(c).lower() for c in df.columns.get_level_values(0)]
        df.columns = df.columns.get_level_values(0 if any(k in l0 for k in ['close', 'open', 'high', 'low']) else 1)
    df.columns = [str(c).lower() for c in df.columns]
    req = ['open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in req): return None
    df = df[req].dropna(subset=['open', 'high', 'low', 'close']).copy()
    return df if len(df) >= 15 else None

def calc_atr(df):
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

def compute_indicators(df):
    df = clean_cols(df)
    if df is None or len(df) < 130: return None
    df['high_125'] = df['high'].shift(1).rolling(125).max()
    df['vol_sma_125'] = df['volume'].rolling(125).mean()
    d = df['close'].diff()
    ag = d.where(d > 0, 0.0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = (-d.where(d < 0, 0.0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['rsi'] = 100.0 - (100.0 / (1.0 + (ag / (al + 1e-9))))
    df['atr'] = calc_atr(df)
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
    fig.update_layout(title=title, height=height, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=35, b=15, l=15, r=15))
    return fig

def render_hud_box(title, bg, border, price, fail_str, ladder):
    badges = "".join([f"<div style='background:rgba(255,255,255,0.08);padding:6px 10px;border-radius:6px;min-width:120px;margin:2px;display:inline-block;'><div style='color:#B0B0B0;font-size:11px;'>{n}</div><div style='color:#FFF;font-weight:bold;font-size:15px;'>{l:,.2f}</div></div>" for n, l in ladder])
    st.markdown(f"<div style='background-color:{bg};border:2px solid {border};border-radius:8px;padding:12px;margin-bottom:10px;'><h3 style='color:#FFF;margin:0 0 4px 0;'>{title}</h3><div style='color:#DDD;font-size:12px;margin-bottom:6px;'>Close: <b>{price:,.2f}</b> | Fail LVL: <b style='color:#FFAA00;'>{fail_str}</b></div><div style='display:flex;flex-wrap:wrap;gap:4px;'>{badges}</div></div>", unsafe_allow_html=True)

# ========================================================
# TAB 1: GOLD (XAU/USD) DASHBOARD
# ========================================================
with tab1:
    st.header("🥇 Gold (XAU/USD) Trading Terminal")
    
    # Top Control Bar with Recalculate Button
    cg1, cg2 = st.columns([3.5, 1.2])
    with cg1:
        gold_strategy = st.radio(
            "🎯 Choose Gold Strategy:",
            ("Strategy 3: ULTRA V1.2 (3-Tier TP + Win-Rate)", "Strategy 2: 30M VWAP + 9 EMA + PDH/PDL", "Strategy 1: 125-Bar Breakout"),
            horizontal=True
        )
    with cg2:
        st.write("")
        if st.button("🔄 Recalculate Gold", type="primary", key="btn_recalc_gold"):
            st.cache_data.clear(); st.rerun()

    st.divider()

    # 1. ULTRA V1.2
    if "Strategy 3" in gold_strategy:
        st.subheader("⚡ Gold Strategy 3: ★★★ ULTRA_V1.2 ★★★")
        u_tf = st.radio("⏱️ Gold Ultra Timeframe:", ("5-Minute (5M)", "15-Minute (15M)", "30-Minute (30M)", "1-Hour (1H)"), horizontal=True, index=1)
        u_cfg = {"5-Minute (5M)": ("5m", "5d"), "15-Minute (15M)": ("15m", "30d"), "30-Minute (30M)": ("30m", "30d"), "1-Hour (1H)": ("1h", "60d")}[u_tf]

        @st.cache_data(ttl=120)
        def fetch_gold_u(iv, pr):
            df = clean_cols(yf.download("GC=F", period=pr, interval=iv, progress=False))
            if df is None or len(df) < 30: return None
            df['atr'] = calc_atr(df)
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['swing_high'] = df['high'].rolling(5).max()
            df['swing_low'] = df['low'].rolling(5).min()
            return df.dropna(subset=['atr', 'ema_9', 'ema_21'])

        gdf_u = fetch_gold_u(u_cfg[0], u_cfg[1])
        if gdf_u is not None and len(gdf_u) >= 20:
            lb = gdf_u.iloc[-1]
            cp, ca, ce9, ce21 = float(lb['close']), float(lb['atr']), float(lb['ema_9']), float(lb['ema_21'])
            is_s = (cp < ce9) and (ce9 < ce21)
            sig, clr = ("SELL ★", "#FF4D4D") if is_s else ("BUY ★", "#00FF66")
            sl_v = round(float(lb['swing_high']) + 0.5, 2) if is_s else round(float(lb['swing_low']) - 0.5, 2)
            rp = abs(sl_v - cp)
            tp1, tp2, tp3 = (round(cp - 0.5*rp, 2), round(cp - 1.0*rp, 2), round(cp - 1.5*rp, 2)) if is_s else (round(cp + 0.5*rp, 2), round(cp + 1.0*rp, 2), round(cp + 1.5*rp, 2))

            # Fast Backtest (150 bars)
            edf = gdf_u.iloc[-150:].copy()
            t1_h, t2_h, t3_h, sl_h, t_cnt = 0, 0, 0, 0, 0
            for i in range(15, len(edf)-5):
                r_i = edf.iloc[i]
                nxt = edf.iloc[i+1:min(i+15, len(edf))]
                if r_i['ema_9'] > r_i['ema_21'] and edf.iloc[i-1]['ema_9'] <= edf.iloc[i-1]['ema_21']:
                    en, s_lvl = r_i['close'], r_i['swing_low']
                    r_val = max(en - s_lvl, r_i['atr'])
                    t_cnt += 1
                    if nxt['low'].min() <= s_lvl: sl_h += 1
                    else:
                        if nxt['high'].max() >= en + 0.5*r_val: t1_h += 1
                        if nxt['high'].max() >= en + 1.0*r_val: t2_h += 1
                        if nxt['high'].max() >= en + 1.5*r_val: t3_h += 1
                elif r_i['ema_9'] < r_i['ema_21'] and edf.iloc[i-1]['ema_9'] >= edf.iloc[i-1]['ema_21']:
                    en, s_lvl = r_i['close'], r_i['swing_high']
                    r_val = max(s_lvl - en, r_i['atr'])
                    t_cnt += 1
                    if nxt['high'].max() >= s_lvl: sl_h += 1
                    else:
                        if nxt['low'].min() <= en - 0.5*r_val: t1_h += 1
                        if nxt['low'].min() <= en - 1.0*r_val: t2_h += 1
                        if nxt['low'].min() <= en - 1.5*r_val: t3_h += 1

            tc = max(t_cnt, 1)
            p1, p2, p3, ps = round(t1_h*100/tc), round(t2_h*100/tc), round(t3_h*100/tc), max(0, 100 - round(t1_h*100/tc))

            c_u1, c_u2 = st.columns([1.5, 1.2])
            with c_u1:
                st.markdown(f"<div style='background:#131722;border:2px solid {clr};border-radius:8px;padding:14px;'><h2 style='color:{clr};margin:0;'>{sig} (Active)</h2><div style='color:#DDD;font-size:13px;margin:2px 0 8px 0;'>Live Gold: <b>${cp:,.2f}</b></div><div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;'><div style='background:#2A1418;border:1px solid #FF4D4D;padding:6px;border-radius:4px;'><div style='color:#AAA;font-size:10px;'>STOP LOSS</div><div style='color:#FF4D4D;font-weight:bold;'>${sl_v:,.2f}</div></div><div style='background:#112C1E;border:1px solid #00FF66;padding:6px;border-radius:4px;'><div style='color:#AAA;font-size:10px;'>TARGET 1</div><div style='color:#00FF66;font-weight:bold;'>${tp1:,.2f}</div></div><div style='background:#112C1E;border:1px solid #00FF66;padding:6px;border-radius:4px;'><div style='color:#AAA;font-size:10px;'>TARGET 2</div><div style='color:#00FF66;font-weight:bold;'>${tp2:,.2f}</div></div><div style='background:#112C1E;border:1px solid #00FF66;padding:6px;border-radius:4px;'><div style='color:#AAA;font-size:10px;'>TARGET 3</div><div style='color:#00FF66;font-weight:bold;'>${tp3:,.2f}</div></div></div></div>", unsafe_allow_html=True)
            with c_u2:
                st.markdown(f"<div style='background:#250f38;border:2px solid #8e44ad;border-radius:8px;padding:12px;font-family:monospace;'><div style='text-align:center;color:#e0b0ff;font-weight:bold;margin-bottom:6px;'>★★★ ULTRA_V1.2 ★★★</div><div style='display:flex;justify-content:space-between;'><span>TP1: {t1_h}</span><span style='background:#27ae60;padding:1px 6px;border-radius:3px;'>{p1}%</span></div><div style='display:flex;justify-content:space-between;'><span>TP2: {t2_h}</span><span style='background:#16a085;padding:1px 6px;border-radius:3px;'>{p2}%</span></div><div style='display:flex;justify-content:space-between;'><span>TP3: {t3_h}</span><span style='background:#2980b9;padding:1px 6px;border-radius:3px;'>{p3}%</span></div><div style='display:flex;justify-content:space-between;'><span>SL : {sl_h}</span><span style='background:#d35400;padding:1px 6px;border-radius:3px;'>{ps}%</span></div><div style='border-top:1px solid #5b2c6f;margin-top:4px;padding-top:4px;display:flex;justify-content:space-between;'><span style='color:#f39c12;font-weight:bold;'>WIN RATE:</span><span style='color:#2ecc71;font-weight:bold;'>{p1}%</span></div></div>", unsafe_allow_html=True)

            h_u = [(sl_v, '#FF4D4D', 'SL', 'solid'), (cp, '#FFF', 'Entry', 'dot'), (tp1, '#27ae60', 'TP1', 'dash'), (tp2, '#16a085', 'TP2', 'dash'), (tp3, '#2980b9', 'TP3', 'dash')]
            st.plotly_chart(plot_candles(gdf_u.iloc[-90:], f"Gold ULTRA V1.2 ({u_tf})", h_u, [('ema_9', 'cyan', 1.5), ('ema_21', 'magenta', 1.5)], height=540), use_container_width=True)

    # 2. STRATEGY 2: VWAP + 9 EMA + PDH/PDL
    elif "Strategy 2" in gold_strategy:
        st.subheader("⚡ Gold Strategy 2: 30-Minute Session VWAP + 9 EMA + PDH / PDL")

        @st.cache_data(ttl=120)
        def fetch_gold_30m():
            df = clean_cols(yf.download("GC=F", period="30d", interval="30m", progress=False))
            if df is None or len(df) < 50: return None, None, None
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['date'] = df.index.date
            tp = (df['high'] + df['low'] + df['close']) / 3.0
            df['vwap'] = (tp * df['volume']).groupby(df['date']).cumsum() / (df['volume'].groupby(df['date']).cumsum() + 1e-9)
            df['atr'] = calc_atr(df)
            counts = df.groupby('date').size()
            valid_days = counts[counts >= 10].index
            pdh, pdl = (float(df[df['date'] == valid_days[-2]]['high'].max()), float(df[df['date'] == valid_days[-2]]['low'].min())) if len(valid_days) >= 2 else (float(df['high'].max()), float(df['low'].min()))
            return df.dropna(subset=['ema_9', 'vwap', 'atr']), pdh, pdl

        df_30m, pdh_v, pdl_v = fetch_gold_30m()
        if df_30m is not None and len(df_30m) >= 2:
            lb = df_30m.iloc[-1]
            p, v, e9, a = round(float(lb['close']), 2), round(float(lb['vwap']), 2), round(float(lb['ema_9']), 2), round(float(lb['atr']), 2)
            is_b, is_s = (p > v and e9 > v), (p < v and e9 < v)
            sig = "🟢 STRONG BUY" if is_b else ("🔴 STRONG SELL" if is_s else "⚪ NEUTRAL")
            sl = f"${min(v - 1.0, round(p - 1.5 * a, 2)):,.2f}" if is_b else (f"${max(v + 1.0, round(p + 1.5 * a, 2)):,.2f}" if is_s else "-")
            tgt = f"${round(p + 3.0 * a, 2):,.2f}" if is_b else (f"${round(p - 3.0 * a, 2):,.2f}" if is_s else "-")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Live Gold Price", f"${p:,.2f}"); m2.metric("Signal", sig); m3.metric("Stop-Loss", sl); m4.metric("Target (1:2)", tgt)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VWAP", f"${v:,.2f}"); c2.metric("9 EMA", f"${e9:,.2f}"); c3.metric("PDH", f"${pdh_v:,.2f}"); c4.metric("PDL", f"${pdl_v:,.2f}")

            plot_df = df_30m.iloc[-120:]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name="Gold"), 1, 1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['vwap'], line=dict(color='orange', width=2), name="VWAP"), 1, 1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_9'], line=dict(color='cyan', width=1.5), name="9 EMA"), 1, 1)
            fig.add_hline(y=pdh_v, line=dict(color='#00FF66', width=1.5, dash='dash'), annotation_text="PDH", row=1, col=1)
            fig.add_hline(y=pdl_v, line=dict(color='#FF4D4D', width=1.5, dash='dash'), annotation_text="PDL", row=1, col=1)
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], name="Volume"), 2, 1)
            fig.update_layout(height=580, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

    # 3. STRATEGY 1: 125-BAR BREAKOUT
    else:
        st.subheader("🏆 Gold Strategy 1: 125-Bar Breakout (Multi-Timeframe)")
        gold_tf = st.radio("⏱️ Select Timeframe:", ("Daily (1D)", "4-Hour (4H)", "1-Hour (1H)", "15-Minute (15M)"), horizontal=True)
        g_iv, g_pr = {"Daily (1D)": ("1d", "2y"), "4-Hour (4H)": ("4h", "120d"), "1-Hour (1H)": ("1h", "60d"), "15-Minute (15M)": ("15m", "30d")}[gold_tf]

        @st.cache_data(ttl=120)
        def fetch_gold_bo(iv, pr):
            raw = clean_cols(yf.download("GC=F", period=pr, interval=("1h" if iv=="4h" else iv), progress=False))
            df = raw.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna() if iv=="4h" and raw is not None else raw
            return compute_indicators(df)

        gold_df = fetch_gold_bo(g_iv, g_pr)
        if gold_df is not None and len(gold_df) >= 2:
            lb = gold_df.iloc[-1]
            p, h125, a = round(float(lb['close']), 2), round(float(lb['high_125']), 2), round(float(lb['atr']), 2)
            vr = round(float(lb['volume']) / (float(lb['vol_sma_125']) + 1e-9), 2)
            is_b, is_s = (p >= h125 and vr >= 1.0), (lb['ema_21'] < lb['ema_50'])
            sig = "🟢 STRONG BUY" if is_b else ("🔴 SELL" if is_s else "⚪ NEUTRAL")
            sl = f"${p - 2*a:,.2f}" if is_b else (f"${p + 2*a:,.2f}" if is_s else "-")
            tgt = f"${p + 4*a:,.2f}" if is_b else (f"${p - 4*a:,.2f}" if is_s else "-")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Live Gold", f"${p:,.2f}"); m2.metric("Signal", sig); m3.metric("Stop-Loss (2 ATR)", sl); m4.metric("Target (4 ATR)", tgt)
            st.plotly_chart(plot_candles(gold_df.iloc[-100:], f"Gold Breakout ({gold_tf})", [(h125, 'orange', '125 High', 'dash')], [('ema_21', 'blue', 1.5), ('ema_50', 'red', 1.5)]), use_container_width=True)

# ========================================================
# TAB 2: INTRADAY OPTION MOMENTUM & VOLATILITY EXTENSION
# ========================================================
with tab2:
    st.header("🎯 Intraday Option Momentum & Volatility Extension System")
    st.caption("Multi-Timeframe Engine: Independent Signals, Fail Levels & 5-Tier Target Ladders on 15M, 1H, Daily & Monthly")

    PRESETS = {
        "NIFTY 50 (NSE)": "^NSEI", "BANK NIFTY (NSE)": "^NSEBANK", "FIN NIFTY (NSE)": "NIFTY_FIN_SERVICE.NS",
        "MIDCP NIFTY (NSE)": "NIFTY_MID_SELECT.NS", "SENSEX (BSE)": "^BSESN", "BANKEX (BSE)": "BSE-BANK.BO",
        "COMEX Gold": "GC=F", "Custom": "CUSTOM"
    }

    cs1, cs2, cs3 = st.columns([1.6, 1.4, 1.0])
    with cs1: chosen = st.selectbox("Select Asset / Index:", list(PRESETS.keys()))
    with cs2: u_sym = st.text_input("Ticker Symbol:", value="RELIANCE.NS" if chosen == "Custom" else PRESETS[chosen], disabled=(chosen != "Custom")).strip().upper()
    with cs3:
        st.write("")
        if st.button("🔄 Recalculate Multi-TF", type="primary", key="btn_recalc_opt"):
            st.cache_data.clear(); st.rerun()

    @st.cache_data(ttl=60)
    def fetch_m_tf(sym):
        d15 = clean_cols(yf.download(sym, period="30d", interval="15m", progress=False))
        d1h = clean_cols(yf.download(sym, period="60d", interval="1h", progress=False))
        d1d = clean_cols(yf.download(sym, period="2y", interval="1d", progress=False))
        d1m = clean_cols(yf.download(sym, period="10y", interval="1mo", progress=False))
        out = []
        for d in [d15, d1h, d1d, d1m]:
            if d is not None and len(d) >= 15:
                d['atr'] = calc_atr(d)
                d['ema_9'] = d['close'].ewm(span=9, adjust=False).mean()
                d['ema_21'] = d['close'].ewm(span=21, adjust=False).mean()
                d['ema_50'] = d['close'].ewm(span=50, adjust=False).mean()
            out.append(d)
        return out[0], out[1], out[2], out[3]

    with st.spinner(f"Loading Multi-Timeframe feeds for {u_sym}..."):
        d15, d1h, d1d, d1m = fetch_m_tf(u_sym)

    def render_opt_tf(df, tf_name, extra_h=None, extra_e=None, bars=90):
        if df is None or df.empty or len(df) < 15:
            st.warning(f"Insufficient data for {tf_name}."); return

        r = df.iloc[-1]
        c_p = float(r['close'])
        c_a = float(r['atr']) if ('atr' in r and not np.isnan(r['atr'])) else (c_p * 0.01)
        c_e9, c_e21 = float(r['ema_9']), float(r['ema_21'])

        is_bull, is_bear = (c_p >= c_e9 and c_e9 >= c_e21), (c_p < c_e9 and c_e9 < c_e21)
        step = round(c_a * 1.5, 2)
        h_step = round(step / 2.0, 2)

        if is_bull:
            badge, title, bg, border = "🟢 BUY (LONG / CALL)", f"CONTINUE VIEW UP — 🟢 BUY ACTIVE ({tf_name})", "#0d3b1e", "#00FF66"
            f_lvl, f_str = round(c_p - step, 2), f"{round(c_p - step, 2):,.2f} (-{step:,.2f} pts)"
            ladder = [(f"Target {i+1}", round(c_p + step + i*h_step, 2)) for i in range(5)]
        elif is_bear:
            badge, title, bg, border = "🔴 SELL (SHORT / PUT)", f"CONTINUE VIEW DOWN — 🔴 SELL ACTIVE ({tf_name})", "#421616", "#FF4D4D"
            f_lvl, f_str = round(c_p + step, 2), f"{round(c_p + step, 2):,.2f} (+{step:,.2f} pts)"
            ladder = [(f"Target {i+1}", round(c_p - step - i*h_step, 2)) for i in range(5)]
        else:
            badge, title, bg, border = "⚪ NEUTRAL / WAIT", f"CONSOLIDATION — ⚪ NO SIGNAL ({tf_name})", "#222222", "#888888"
            f_lvl, f_str = round(c_p - step, 2), f"{round(c_p - step, 2):,.2f} ({step:,.2f} pts)"
            ladder = [(f"Range {i+1}", round(c_p + step + i*h_step, 2)) for i in range(5)]

        # KPI Metrics Row
        k1, k2, k3, k4, k5 = st.columns([1.5, 1.2, 1.3, 1.2, 1.2])
        k1.metric("Signal", badge); k2.metric("Current Price", f"{c_p:,.2f}"); k3.metric("Stop Loss (Fail LVL)", f"{f_lvl:,.2f}")
        k4.metric("Target 1", f"{ladder[0][1]:,.2f}"); k5.metric("Target 3", f"{ladder[2][1]:,.2f}")

        # HUD Box
        render_hud_box(title, bg, border, c_p, f_str, ladder)

        # Plotly Candlestick Chart
        h_lines = [(f_lvl, '#FFAA00', 'Fail LVL', 'dot')] + [(l, '#00FF66' if is_bull else '#FF4D4D', n, 'dash') for n, l in ladder]
        if extra_h: h_lines.extend(extra_h)
        emas = [('ema_9', 'cyan', 1.5), ('ema_21', 'magenta', 1.5)]
        if extra_e: emas.extend(extra_e)

        st.plotly_chart(plot_candles(df.iloc[-bars:], f"{tf_name} Targets ({u_sym})", h_lines, emas, height=540), use_container_width=True)

    ct1, ct2, ct3, ct4 = st.tabs(["⚡ 15-Minute", "⏰ 1-Hour", "📅 Daily", "🏛️ Monthly"])
    with ct1: render_opt_tf(d15, "15-Minute Execution", bars=90)
    with ct2: render_opt_tf(d1h, "1-Hour Swing", bars=90)
    with ct3:
        ex_d = [(float(d1d.iloc[-2]['high']), '#00FF66', 'PDH', 'dash'), (float(d1d.iloc[-2]['low']), '#FF4D4D', 'PDL', 'dash')] if d1d is not None and len(d1d) >= 2 else []
        render_opt_tf(d1d, "Daily Positional", extra_h=ex_d, extra_e=[('ema_50', 'orange', 1.5)], bars=120)
    with ct4:
        ex_m = [(float(d1m.iloc[-2]['high']), '#00FF66', 'PMH', 'dash'), (float(d1m.iloc[-2]['low']), '#FF4D4D', 'PML', 'dash')] if d1m is not None and len(d1m) >= 2 else []
        render_opt_tf(d1m, "Monthly Macro", extra_h=ex_m, bars=48)

# === END OF APY.PY ===
