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
# 1. CORE UTILITIES & INDICATORS
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
    df['low_125'] = df['low'].shift(1).rolling(125).min()
    df['vol_sma_125'] = df['volume'].rolling(125).mean()
    d = df['close'].diff()
    ag = d.where(d > 0, 0.0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = (-d.where(d < 0, 0.0)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df['rsi'] = 100.0 - (100.0 / (1.0 + (ag / (al + 1e-9))))
    df['atr'] = calc_atr(df)
    df['atr_sma_50'] = df['atr'].rolling(min(50, len(df)-1)).mean()
    df['sma_50'] = df['close'].rolling(min(50, len(df)-1)).mean()
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
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
# TAB 1: GOLD (XAU/USD) DASHBOARD (STRATEGIES 1, 2, AND 3)
# ========================================================
with tab1:
    gold_strategy = st.radio(
        "🎯 Choose Gold Strategy:",
        (
            "Strategy 3: ULTRA V1.2 (3-Tier TP + Live Win-Rate Engine)",
            "Strategy 2: 30M Session VWAP + 9 EMA + PDH/PDL",
            "Strategy 1: 125-Bar Breakout + Volume (Multi-Timeframe)"
        ),
        horizontal=True
    )

    # ----------------------------------------------------
    # STRATEGY 3: ULTRA V1.2 (3-TIER TP + WIN RATE)
    # ----------------------------------------------------
    if "Strategy 3" in gold_strategy:
        st.subheader("⚡ Gold Strategy 3: ★★★ ULTRA_V1.2 ★★★")
        st.caption("Rejection Candlestick + SuperTrend Momentum with SL and 3-Tier Take-Profit Ladder (TP1, TP2, TP3) & Live Win Rate Engine")
        u_tf = st.radio("⏱️ Gold Ultra Timeframe:", ("5-Minute (5M)", "15-Minute (15M)", "30-Minute (30M)", "1-Hour (1H)"), horizontal=True, index=1)
        u_cfg = {"5-Minute (5M)": ("5m", "5d"), "15-Minute (15M)": ("15m", "30d"), "30-Minute (30M)": ("30m", "30d"), "1-Hour (1H)": ("1h", "60d")}[u_tf]

        @st.cache_data(ttl=120)
        def fetch_gold_ultra(iv, pr):
            df = clean_cols(yf.download("GC=F", period=pr, interval=iv, progress=False))
            if df is None or len(df) < 30: return None
            df['atr'] = calc_atr(df)
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['swing_high'] = df['high'].rolling(5).max()
            df['swing_low'] = df['low'].rolling(5).min()
            return df.dropna(subset=['atr', 'ema_9', 'ema_21'])

        gdf_u = fetch_gold_ultra(u_cfg[0], u_cfg[1])
        if gdf_u is not None and len(gdf_u) >= 20:
            last_bar = gdf_u.iloc[-1]
            c_p, c_a = float(last_bar['close']), float(last_bar['atr'])
            c_e9, c_e21 = float(last_bar['ema_9']), float(last_bar['ema_21'])
            is_short = (c_p < c_e9) and (c_e9 < c_e21)

            if is_short:
                sig_type, clr = "SELL ★", "#FF4D4D"
                sl_val = round(float(last_bar['swing_high']) + 0.5, 2)
                r_pts = abs(sl_val - c_p)
                tp1_val = round(c_p - (0.5 * r_pts), 2)
                tp2_val = round(c_p - (1.0 * r_pts), 2)
                tp3_val = round(c_p - (1.5 * r_pts), 2)
            else:
                sig_type, clr = "BUY ★", "#00FF66"
                sl_val = round(float(last_bar['swing_low']) - 0.5, 2)
                r_pts = abs(c_p - sl_val)
                tp1_val = round(c_p + (0.5 * r_pts), 2)
                tp2_val = round(c_p + (1.0 * r_pts), 2)
                tp3_val = round(c_p + (1.5 * r_pts), 2)

            eval_df = gdf_u.iloc[-150:].copy()
            tp1_hits, tp2_hits, tp3_hits, sl_hits, total_eval = 0, 0, 0, 0, 0
            for i in range(15, len(eval_df)-5):
                row = eval_df.iloc[i]
                nxt = eval_df.iloc[i+1:min(i+15, len(eval_df))]
                if row['ema_9'] > row['ema_21'] and eval_df.iloc[i-1]['ema_9'] <= eval_df.iloc[i-1]['ema_21']:
                    entry, sl = row['close'], row['swing_low']
                    r = max(entry - sl, row['atr'])
                    total_eval += 1
                    if nxt['low'].min() <= sl: sl_hits += 1
                    else:
                        if nxt['high'].max() >= entry + 0.5*r: tp1_hits += 1
                        if nxt['high'].max() >= entry + 1.0*r: tp2_hits += 1
                        if nxt['high'].max() >= entry + 1.5*r: tp3_hits += 1
                elif row['ema_9'] < row['ema_21'] and eval_df.iloc[i-1]['ema_9'] >= eval_df.iloc[i-1]['ema_21']:
                    entry, sl = row['close'], row['swing_high']
                    r = max(sl - entry, row['atr'])
                    total_eval += 1
                    if nxt['high'].max() >= sl: sl_hits += 1
                    else:
                        if nxt['low'].min() <= entry - 0.5*r: tp1_hits += 1
                        if nxt['low'].min() <= entry - 1.0*r: tp2_hits += 1
                        if nxt['low'].min() <= entry - 1.5*r: tp3_hits += 1

            t_cnt = max(total_eval, 1)
            tp1_pct = round((tp1_hits / t_cnt) * 100)
            tp2_pct = round((tp2_hits / t_cnt) * 100)
            tp3_pct = round((tp3_hits / t_cnt) * 100)
            sl_pct = max(0, 100 - tp1_pct)

            c_u1, c_u2 = st.columns([1.5, 1.2])
            with c_u1:
                st.markdown(f"<div style='background:#131722;border:2px solid {clr};border-radius:10px;padding:16px;'><h2 style='color:{clr};margin:0;'>{sig_type} SIGNAL (Active)</h2><div style='color:#DDD;font-size:13px;margin:4px 0 10px 0;'>Live Price: <b>${c_p:,.2f}</b></div><div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'><div style='background:#2A1418;border:1px solid #FF4D4D;padding:8px;border-radius:6px;'><div style='color:#AAA;font-size:11px;'>STOP LOSS (SL)</div><div style='color:#FF4D4D;font-size:17px;font-weight:bold;'>${sl_val:,.2f}</div></div><div style='background:#112C1E;border:1px solid #00FF66;padding:8px;border-radius:6px;'><div style='color:#AAA;font-size:11px;'>TARGET 1 (TP1)</div><div style='color:#00FF66;font-size:17px;font-weight:bold;'>${tp1_val:,.2f}</div></div><div style='background:#112C1E;border:1px solid #00FF66;padding:8px;border-radius:6px;'><div style='color:#AAA;font-size:11px;'>TARGET 2 (TP2)</div><div style='color:#00FF66;font-size:17px;font-weight:bold;'>${tp2_val:,.2f}</div></div><div style='background:#112C1E;border:1px solid #00FF66;padding:8px;border-radius:6px;'><div style='color:#AAA;font-size:11px;'>TARGET 3 (TP3)</div><div style='color:#00FF66;font-size:17px;font-weight:bold;'>${tp3_val:,.2f}</div></div></div></div>", unsafe_allow_html=True)
            with c_u2:
                st.markdown(f"<div style='background:#250f38;border:2px solid #8e44ad;border-radius:10px;padding:14px;font-family:monospace;'><div style='text-align:center;color:#e0b0ff;font-weight:bold;margin-bottom:8px;'>★★★ ULTRA_V1.2 ★★★</div><div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='color:#FFF;'>TP1: {tp1_hits}</span><span style='background:#27ae60;color:#FFF;padding:1px 6px;border-radius:4px;font-weight:bold;'>{tp1_pct}%</span></div><div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='color:#FFF;'>TP2: {tp2_hits}</span><span style='background:#16a085;color:#FFF;padding:1px 6px;border-radius:4px;font-weight:bold;'>{tp2_pct}%</span></div><div style='display:flex;justify-content:space-between;margin-bottom:4px;'><span style='color:#FFF;'>TP3: {tp3_hits}</span><span style='background:#2980b9;color:#FFF;padding:1px 6px;border-radius:4px;font-weight:bold;'>{tp3_pct}%</span></div><div style='display:flex;justify-content:space-between;margin-bottom:8px;'><span style='color:#FFF;'>SL : {sl_hits}</span><span style='background:#d35400;color:#FFF;padding:1px 6px;border-radius:4px;font-weight:bold;'>{sl_pct}%</span></div><div style='border-top:1px solid #5b2c6f;padding-top:6px;display:flex;justify-content:space-between;'><span style='color:#f39c12;font-weight:bold;'>🔥 WIN RATE:</span><span style='color:#2ecc71;font-weight:bold;font-size:16px;'>{tp1_pct}%</span></div></div>", unsafe_allow_html=True)

            h_ultra = [(sl_val, '#FF4D4D', 'SL', 'solid'), (c_p, '#FFFFFF', 'Entry', 'dot'), (tp1_val, '#27ae60', 'TP1', 'dash'), (tp2_val, '#16a085', 'TP2', 'dash'), (tp3_val, '#2980b9', 'TP3', 'dash')]
            st.plotly_chart(plot_candles(gdf_u.iloc[-90:], f"Gold ULTRA V1.2 ({u_tf})", h_ultra, [('ema_9', 'cyan', 1.5), ('ema_21', 'magenta', 1.5)], height=560), use_container_width=True)

    # ----------------------------------------------------
    # STRATEGY 2: VWAP + 9 EMA + PDH/PDL
    # ----------------------------------------------------
    elif "Strategy 2" in gold_strategy:
        st.subheader("⚡ Gold Strategy 2: 30-Minute Session VWAP + 9 EMA + PDH / PDL")
        if st.button("🔄 Refresh 30M Gold Strategy", type="primary"):
            st.cache_data.clear(); st.rerun()

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
            if len(valid_days) >= 2:
                prev_df = df[df['date'] == valid_days[-2]]
                pdh, pdl = float(prev_df['high'].max()), float(prev_df['low'].min())
            else:
                pdh, pdl = float(df['high'].max()), float(df['low'].min())
            return df.dropna(subset=['ema_9', 'vwap', 'atr']), pdh, pdl

        df_30m, pdh_val, pdl_val = fetch_gold_30m()
        if df_30m is not None and len(df_30m) >= 2:
            latest = df_30m.iloc[-1]
            p, v, e9, a = round(float(latest['close']), 2), round(float(latest['vwap']), 2), round(float(latest['ema_9']), 2), round(float(latest['atr']), 2)
            is_bull, is_bear = (p > v) and (e9 > v), (p < v) and (e9 < v)
            pd_status = "🚀 Above PDH Breakout" if p > pdh_val else ("⚠️ Below PDL Breakdown" if p < pdl_val else "↔️ Inside Range")
            sig = "🟢 STRONG BUY" if is_bull else ("🔴 STRONG SELL" if is_bear else "⚪ NEUTRAL")
            sl = f"${min(v - 1.0, round(p - 1.5 * a, 2)):,.2f}" if is_bull else (f"${max(v + 1.0, round(p + 1.5 * a, 2)):,.2f}" if is_bear else "-")
            tgt = f"${round(p + 3.0 * a, 2):,.2f}" if is_bull else (f"${round(p - 3.0 * a, 2):,.2f}" if is_bear else "-")

            m1, m2, m3, m4 = st.columns([1.2, 2.0, 1.4, 1.4])
            m1.metric("Live Gold Price (30M)", f"${p:,.2f}"); m2.metric("Signal", sig); m3.metric("Stop-Loss", sl); m4.metric("Target (1:2 RR)", tgt)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Session VWAP", f"${v:,.2f}"); c2.metric("9 EMA", f"${e9:,.2f}"); c3.metric("PDH", f"${pdh_val:,.2f}"); c4.metric("PDL", f"${pdl_val:,.2f}")
            st.info(f"**PDH/PDL Context:** {pd_status}")

            plot_df = df_30m.iloc[-150:]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name="Gold (30M)"), 1, 1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['vwap'], line=dict(color='orange', width=2.5), name="Session VWAP"), 1, 1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_9'], line=dict(color='cyan', width=1.8), name="9 EMA"), 1, 1)
            fig.add_hline(y=pdh_val, line=dict(color='#00FF66', width=2, dash='dash'), annotation_text=f"PDH: ${pdh_val:,.2f}", row=1, col=1)
            fig.add_hline(y=pdl_val, line=dict(color='#FF4D4D', width=2, dash='dash'), annotation_text=f"PDL: ${pdl_val:,.2f}", row=1, col=1)
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], name="COMEX Volume"), 2, 1)
            fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # STRATEGY 1: 125-BAR BREAKOUT
    # ----------------------------------------------------
    else:
        st.subheader("🏆 Gold Strategy 1: 125-Bar Breakout + Volume (Multi-Timeframe)")
        gold_tf = st.radio("⏱️ Select Timeframe:", ("Daily (1D)", "4-Hour (4H)", "1-Hour (1H)", "15-Minute (15M)"), horizontal=True)
        tf_map = {"Daily (1D)": ("1d", "2y", "125-Day"), "4-Hour (4H)": ("4h", "120d", "125-Bar (4H)"), "1-Hour (1H)": ("1h", "60d", "125-Hour"), "15-Minute (15M)": ("15m", "30d", "125-Bar (15M)")}
        g_iv, g_pr, g_lbl = tf_map[gold_tf]

        @st.cache_data(ttl=120)
        def fetch_gold_bo(iv, pr):
            if iv == "4h":
                raw = clean_cols(yf.download("GC=F", period=pr, interval="1h", progress=False))
                df = raw.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna() if raw is not None else None
            else:
                df = yf.download("GC=F", period=pr, interval=iv, progress=False)
            return compute_indicators(df)

        gold_df = fetch_gold_bo(g_iv, g_pr)
        if gold_df is not None and len(gold_df) >= 2:
            row = gold_df.iloc[-1]
            p, h125, a = round(float(row['close']), 2), round(float(row['high_125']), 2), round(float(row['atr']), 2)
            vr = round(float(row['volume']) / (float(row['vol_sma_125']) + 1e-9), 2)
            is_buy = (p >= h125) and (vr >= 1.0)
            is_sell = row['ema_21'] < row['ema_50']
            sig = "🟢 STRONG BUY" if is_buy else ("🔴 SELL / SHORT" if is_sell else "⚪ NEUTRAL")
            sl = f"${round(p - 2*a, 2):,.2f}" if is_buy else (f"${round(p + 2*a, 2):,.2f}" if is_sell else "-")
            tgt = f"${round(p + 4*a, 2):,.2f}" if is_buy else (f"${round(p - 4*a, 2):,.2f}" if is_sell else "-")

            m1, m2, m3, m4 = st.columns([1.2, 2.0, 1.4, 1.4])
            m1.metric("Live Gold Price", f"${p:,.2f}"); m2.metric("Signal", sig); m3.metric("Stop-Loss (2 ATR)", sl); m4.metric("Target (4 ATR)", tgt)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=gold_df.index, open=gold_df['open'], high=gold_df['high'], low=gold_df['low'], close=gold_df['close'], name="Price"), 1, 1)
            fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['high_125'], line=dict(color='orange', dash='dash'), name=f"{g_lbl} High"), 1, 1)
            fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['ema_21'], line=dict(color='blue', width=1.5), name="21 EMA"), 1, 1)
            fig.add_trace(go.Scatter(x=gold_df.index, y=gold_df['ema_50'], line=dict(color='red', width=1.5), name="50 EMA"), 1, 1)
            fig.add_trace(go.Bar(x=gold_df.index, y=gold_df['volume'], name="Volume"), 2, 1)
            fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

# ========================================================
# TAB 2: INTRADAY OPTION MOMENTUM & MULTI-TIMEFRAME ENGINE
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
    with cs2:
        u_sym = st.text_input("Ticker Symbol:", value="RELIANCE.NS" if chosen == "Custom" else PRESETS[chosen], disabled=(chosen != "Custom")).strip().upper()
    with cs3:
        st.write(""); st.write("")
        if st.button("🔄 Recalculate Multi-TF", type="primary"):
            st.cache_data.clear(); st.rerun()

    @st.cache_data(ttl=60)
    def fetch_m_tf(sym):
        d15 = clean_cols(yf.download(sym, period="30d", interval="15m", progress=False))
        d1h = clean_cols(yf.download(sym, period="60d", interval="1h", progress=False))
        d1d = clean_cols(yf.download(sym, period="2y", interval="1d", progress=False))
        d1m = clean_cols(yf.download(sym, period="10y", interval="1mo", progress=False))
        
        results = []
        for d in [d15, d1h, d1d, d1m]:
            if d is not None and len(d) >= 15:
                d['atr'] = calc_atr(d)
                d['ema_9'] = d['close'].ewm(span=9, adjust=False).mean()
                d['ema_21'] = d['close'].ewm(span=21, adjust=False).mean()
                d['ema_50'] = d['close'].ewm(span=50, adjust=False).mean()
            results.append(d)
        return results[0], results[1], results[2], results[3]

    with st.spinner(f"Loading 15M, 1H, Daily, and Monthly feeds for {u_sym}..."):
        d15, d1h, d1d, d1m = fetch_m_tf(u_sym)

    def render_opt_tf(df, tf_name, extra_h=None, extra_e=None, bars=90):
        if df is None or df.empty or len(df) < 15:
            st.warning(f"Insufficient data for {tf_name}."); return

        r = df.iloc[-1]
        c_p = float(r['close'])
        c_a = float(r['atr']) if (('atr' in r) and not np.isnan(r['atr'])) else (c_p * 0.01)
        c_e9, c_e21 = float(r['ema_9']), float(r['ema_21'])

        is_bull = (c_p >= c_e9) and (c_e9 >= c_e21)
        is_bear = (c_p < c_e9) and (c_e9 < c_e21)
        step = round(c_a * 1.5, 2)
        h_step = round(step / 2.0, 2)

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

        # KPI Metrics Row
        k1, k2, k3, k4, k5 = st.columns([1.5, 1.2, 1.3, 1.2, 1.2])
        k1.metric("Signal", badge); k2.metric("Current Price", f"{c_p:,.2f}"); k3.metric("Stop Loss (Fail LVL)", f"{f_lvl:,.2f}")
        k4.metric("Target 1", f"{ladder[0][1]:,.2f}"); k5.metric("Target 3", f"{ladder[2][1]:,.2f}")

        # Visual HUD Card
        badges_html = "".join([f"<div style='background:rgba(255,255,255,0.08);padding:8px 12px;border-radius:6px;min-width:130px;margin:3px;display:inline-block;'><div style='color:#B0B0B0;font-size:12px;'>{n}</div><div style='color:#FFF;font-weight:bold;font-size:16px;'>{l:,.2f}</div></div>" for n, l in ladder])
        st.markdown(f"<div style='background-color:{bg};border:2px solid {border};border-radius:10px;padding:14px;margin-bottom:12px;'><h3 style='color:#FFF;margin:0;'>{title}</h3><div style='color:#E0E0E0;font-size:13px;margin:3px 0 8px 0;'>Current Close: <b>{c_p:,.2f}</b> | Mode: <b>{act}</b></div><div style='background:rgba(0,0,0,0.35);padding:6px 12px;border-radius:6px;margin-bottom:8px;'><span style='color:#FFAA00;font-weight:bold;'>Move View Fail LVL: </span><span style='color:#FFF;font-weight:bold;font-size:16px;'>{f_str}</span></div><div style='display:flex;flex-wrap:wrap;gap:4px;'>{badges_html}</div></div>", unsafe_allow_html=True)

        h_lines = [(f_lvl, '#FFAA00', 'Fail LVL', 'dot')] + [(l, '#00FF66' if is_bull else '#FF4D4D', n, 'dash') for n, l in ladder]
        if extra_h: h_lines.extend(extra_h)
        emas = [('ema_9', 'cyan', 1.5), ('ema_21', 'magenta', 1.5)]
        if extra_e: emas.extend(extra_e)

        st.plotly_chart(plot_candles(df.iloc[-bars:], f"{tf_name} Targets ({u_sym})", h_lines, emas, height=540), use_container_width=True)

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
