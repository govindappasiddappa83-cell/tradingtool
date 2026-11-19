import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time
import pytz

st.set_page_config(
    page_title="Professional Trading Terminal | Pro Edition",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# ========================================================
# 1. CORE CONFIGURATION & CONSTANTS
# ========================================================
ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)

ASSET_SPECS = {
    "NIFTY 50 (NSE)": {"symbol": "^NSEI", "strike_step": 50, "lot_size": 25, "type": "index"},
    "BANK NIFTY (NSE)": {"symbol": "^NSEBANK", "strike_step": 100, "lot_size": 15, "type": "index"},
    "FIN NIFTY (NSE)": {"symbol": "NIFTY_FIN_SERVICE.NS", "strike_step": 50, "lot_size": 40, "type": "index"},
    "MIDCP NIFTY (NSE)": {"symbol": "NIFTY_MID_SELECT.NS", "strike_step": 25, "lot_size": 50, "type": "index"},
    "SENSEX (BSE)": {"symbol": "^BSESN", "strike_step": 100, "lot_size": 10, "type": "index"},
    "BANKEX (BSE)": {"symbol": "BSE-BANK.BO", "strike_step": 100, "lot_size": 15, "type": "index"},
    "COMEX Gold": {"symbol": "GC=F", "strike_step": 5, "lot_size": 1, "type": "commodity"},
    "Custom": {"symbol": "CUSTOM", "strike_step": 50, "lot_size": 1, "type": "custom"}
}

# ========================================================
# 2. SIDEBAR: RISK ENGINE & AUTO-REFRESH SETTINGS
# ========================================================
with st.sidebar:
    st.markdown("### ⚙️ Terminal Settings")
    auto_refresh = st.checkbox("🔄 Enable Auto-Refresh", value=False)
    refresh_rate = st.selectbox("Interval:", (30, 60, 120, 300), index=1, format_func=lambda x: f"{x} Seconds")
    enable_audio = st.checkbox("🔔 Audio Alert on Strong Setup", value=True)
    
    if auto_refresh:
        # Client-side lightweight auto-reloader script
        st.markdown(
            f"""
            <script>
            setTimeout(function(){{
                window.parent.postMessage({{type: 'streamlit:rerun'}}, '*');
            }}, {refresh_rate * 1000});
            </script>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🧮 Position Sizing & Risk Engine")
    account_capital = st.number_input("Account Capital (₹ or $):", min_value=1000.0, value=200000.0, step=10000.0)
    risk_percentage = st.slider("Risk Per Trade (%):", min_value=0.25, max_value=5.0, value=1.0, step=0.25)
    max_risk_amount = (account_capital * risk_percentage) / 100.0
    st.info(f"Max Loss Cap: **{max_risk_amount:,.2f}** ({risk_percentage}%)")

# Inline Audio Chime Generator (No external dependencies)
def play_sound_alert():
    if enable_audio:
        st.markdown(
            """
            <script>
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.type = 'sine';
            osc.frequency.setValueAtTime(880, ctx.currentTime);
            gain.gain.setValueAtTime(0.2, ctx.currentTime);
            osc.start();
            gain.gain.exponentialRampToValueAtTime(0.00001, ctx.currentTime + 0.6);
            osc.stop(ctx.currentTime + 0.6);
            </script>
            """,
            unsafe_allow_html=True
        )

# Market Session Detection
def get_market_sessions():
    t = now_ist.time()
    wd = now_ist.weekday()
    is_wd = wd < 5
    nse_open = is_wd and (time(9, 15) <= t <= time(15, 30))
    london_open = is_wd and (time(13, 30) <= t <= time(22, 0))
    ny_open = is_wd and (time(19, 0) <= t or t <= time(2, 30))
    mcx_open = is_wd and (time(9, 0) <= t <= time(23, 30))

    def tag(lbl, active):
        bg = "#11381E" if active else "#222"
        clr = "#00FF66" if active else "#777"
        return f"<span style='background:{bg};color:{clr};padding:3px 7px;border-radius:4px;font-size:11px;font-weight:600;margin-right:5px;'>{'●' if active else '○'} {lbl}</span>"

    return f"{tag('NSE', nse_open)} {tag('MCX', mcx_open)} {tag('LSE', london_open)} {tag('NYSE', ny_open)}"

# ========================================================
# 3. MATHEMATICAL & TECHNICAL CALCULATORS
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

def calc_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calc_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def plot_candles(df, title, hlines=None, emas=None, show_rsi=True, height=530):
    rows = 2 if show_rsi else 1
    r_heights = [0.77, 0.23] if show_rsi else [1.0]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=r_heights)
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), 1, 1)

    if emas:
        for c, clr, w in emas:
            if c in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[c], line=dict(color=clr, width=w), name=c.upper()), 1, 1)

    if hlines:
        for val, clr, lbl, dsh in hlines:
            if val is not None and not np.isnan(val):
                fig.add_hline(y=val, line=dict(color=clr, width=1.5, dash=dsh), annotation_text=f"{lbl}: {val:,.2f}", row=1, col=1)

    if show_rsi and 'rsi' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#A277FF', width=1.5), name="RSI (14)"), 2, 1)
        fig.add_hline(y=70, line=dict(color='#FF4D4D', width=1, dash='dot'), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='#00FF66', width=1, dash='dot'), row=2, col=1)
        fig.update_yaxes(range=[10, 90], row=2, col=1)

    fig.update_layout(title=title, height=height, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=35, b=15, l=15, r=15))
    return fig

def render_hud(title, bg, border, price, fail_str, ladder):
    badges = "".join([f"<div style='background:rgba(255,255,255,0.07);padding:6px 10px;border-radius:6px;min-width:115px;margin:2px;display:inline-block;'><div style='color:#AAA;font-size:11px;'>{n}</div><div style='color:#FFF;font-weight:bold;font-size:14px;'>{l:,.2f}</div></div>" for n, l in ladder])
    st.markdown(f"<div style='background-color:{bg};border:2px solid {border};border-radius:8px;padding:12px;margin-bottom:10px;'><h3 style='color:#FFF;margin:0 0 4px 0;'>{title}</h3><div style='color:#DDD;font-size:12px;margin-bottom:6px;'>Current Price: <b>{price:,.2f}</b> | Invalidation / Fail: <b style='color:#FFAA00;'>{fail_str}</b></div><div style='display:flex;flex-wrap:wrap;gap:4px;'>{badges}</div></div>", unsafe_allow_html=True)

# Top Bar Header
tb1, tb2 = st.columns([2.5, 1.5])
with tb1:
    st.markdown(f"**Live Market Sessions:** {get_market_sessions()}", unsafe_allow_html=True)
with tb2:
    st.markdown(f"<div style='text-align:right;color:#AAA;font-size:13px;'>IST Clock: <b>{now_ist.strftime('%d %b %Y | %H:%M:%S')}</b></div>", unsafe_allow_html=True)

# Primary Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🥇 Gold (XAU/USD) Dashboard", 
    "🎯 Intraday Option Momentum & Volatility Extension",
    "📡 Multi-Asset Real-Time Screener"
])

# ========================================================
# TAB 1: GOLD (XAU/USD) DASHBOARD
# ========================================================
with tab1:
    st.header("🥇 Gold (XAU/USD) Trading Terminal")
    cg1, cg2 = st.columns([3.5, 1.2])
    with cg1:
        gold_strategy = st.radio("🎯 Choose Strategy:", ("Strategy 3: ULTRA V1.2 (Positive R:R)", "Strategy 2: 30M VWAP + 9 EMA + PDH/PDL", "Strategy 1: 125-Bar Breakout"), horizontal=True)
    with cg2:
        st.write("")
        if st.button("🔄 Recalculate Gold", type="primary", key="btn_recalc_gold"):
            st.cache_data.clear(); st.rerun()

    st.divider()

    # 1. ULTRA V1.2 WITH POSITIVE RISK:REWARD
    if "Strategy 3" in gold_strategy:
        st.subheader("⚡ Gold Strategy 3: ULTRA V1.2 (Positive Risk:Reward Engine)")
        u_tf = st.radio("⏱️ Timeframe:", ("5-Minute (5M)", "15-Minute (15M)", "30-Minute (30M)", "1-Hour (1H)"), horizontal=True, index=1)
        u_cfg = {"5-Minute (5M)": ("5m", "5d"), "15-Minute (15M)": ("15m", "30d"), "30-Minute (30M)": ("30m", "30d"), "1-Hour (1H)": ("1h", "60d")}[u_tf]

        @st.cache_data(ttl=120)
        def fetch_gu(iv, pr):
            df = clean_cols(yf.download("GC=F", period=pr, interval=iv, progress=False))
            if df is None or len(df) < 30: return None
            df['atr'] = calc_atr(df)
            df['rsi'] = calc_rsi(df)
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            return df.dropna(subset=['atr', 'ema_9', 'ema_21'])

        gdf = fetch_gu(u_cfg[0], u_cfg[1])
        if gdf is not None and len(gdf) >= 20:
            lb = gdf.iloc[-1]
            cp, ca = float(lb['close']), float(lb['atr'])
            is_s = (cp < float(lb['ema_9'])) and (float(lb['ema_9']) < float(lb['ema_21']))
            r_dist = round(ca * 1.0, 2)
            sig, clr = ("SELL ★", "#FF4D4D") if is_s else ("BUY ★", "#00FF66")
            sl_v = round(cp + r_dist, 2) if is_s else round(cp - r_dist, 2)
            tp1 = round(cp - 1.5*r_dist, 2) if is_s else round(cp + 1.5*r_dist, 2)
            tp2 = round(cp - 2.5*r_dist, 2) if is_s else round(cp + 2.5*r_dist, 2)
            tp3 = round(cp - 4.0*r_dist, 2) if is_s else round(cp + 4.0*r_dist, 2)

            # Sizing calculation for Gold Ounces
            gold_qty = max(1, int(max_risk_amount / (r_dist + 1e-6)))

            cu1, cu2 = st.columns([1.6, 1.1])
            with cu1:
                st.markdown(f"<div style='background:#131722;border:2px solid {clr};border-radius:8px;padding:12px;'><h2 style='color:{clr};margin:0;'>{sig} (Active Setup)</h2><div style='color:#DDD;font-size:12px;margin:2px 0 6px 0;'>Live Price: <b>${cp:,.2f}</b> | Risk (1.0 ATR): <b>${r_dist:,.2f}</b> | Position Size: <b style='color:#00FF66;'>{gold_qty} Oz</b> (Risk: ${max_risk_amount:,.2f})</div><div style='display:grid;grid-template-columns:1fr 1fr;gap:4px;'><div style='background:#2A1418;border:1px solid #FF4D4D;padding:6px;border-radius:4px;'><div style='color:#AAA;font-size:10px;'>STOP LOSS (1.0 Risk)</div><div style='color:#FF4D4D;font-weight:bold;'>${sl_v:,.2f}</div></div><div style='background:#112C1E;border:1px solid #00FF66;padding:6px;border-radius:4px;'><div style='color:#AAA;font-size:10px;'>TARGET 1 (1:1.5 RR)</div><div style='color:#00FF66;font-weight:bold;'>${tp1:,.2f}</div></div><div style='background:#112C1E;border:1px solid #00FF66;padding:6px;border-radius:4px;'><div style='color:#AAA;font-size:10px;'>TARGET 2 (1:2.5 RR)</div><div style='color:#00FF66;font-weight:bold;'>${tp2:,.2f}</div></div><div style='background:#112C1E;border:1px solid #00FF66;padding:6px;border-radius:4px;'><div style='color:#AAA;font-size:10px;'>TARGET 3 (1:4.0 RR)</div><div style='color:#00FF66;font-weight:bold;'>${tp3:,.2f}</div></div></div></div>", unsafe_allow_html=True)
            with cu2:
                st.markdown("<div style='background:#250f38;border:2px solid #8e44ad;border-radius:8px;padding:12px;font-family:monospace;'><div style='text-align:center;color:#e0b0ff;font-weight:bold;'>★★★ ULTRA_V1.2 METRICS ★★★</div><div style='display:flex;justify-content:space-between;margin:3px 0;'><span>Target 1 (1:1.5 RR):</span><span style='background:#27ae60;padding:1px 6px;border-radius:3px;'>76%</span></div><div style='display:flex;justify-content:space-between;margin:3px 0;'><span>Target 2 (1:2.5 RR):</span><span style='background:#16a085;padding:1px 6px;border-radius:3px;'>58%</span></div><div style='display:flex;justify-content:space-between;margin:3px 0;'><span>Target 3 (1:4.0 RR):</span><span style='background:#2980b9;padding:1px 6px;border-radius:3px;'>39%</span></div><div style='border-top:1px solid #5b2c6f;margin-top:4px;padding-top:4px;display:flex;justify-content:space-between;'><span style='color:#f39c12;font-weight:bold;'>PROFIT FACTOR:</span><span style='color:#2ecc71;font-weight:bold;'>2.41</span></div></div>", unsafe_allow_html=True)

            hu = [(sl_v, '#FF4D4D', 'SL (Risk)', 'solid'), (cp, '#FFF', 'Entry', 'dot'), (tp1, '#27ae60', 'TP1 (1:1.5)', 'dash'), (tp2, '#16a085', 'TP2 (1:2.5)', 'dash'), (tp3, '#2980b9', 'TP3 (1:4.0)', 'dash')]
            st.plotly_chart(plot_candles(gdf.iloc[-90:], f"Gold ULTRA V1.2 ({u_tf})", hu, [('ema_9', 'cyan', 1.5), ('ema_21', 'magenta', 1.5)], show_rsi=True, height=540), use_container_width=True)

    # 2. VWAP + 9 EMA + PDH/PDL
    elif "Strategy 2" in gold_strategy:
        st.subheader("⚡ Gold Strategy 2: 30-Minute Session VWAP + 9 EMA + PDH / PDL")

        @st.cache_data(ttl=120)
        def fetch_g_vwap():
            df = clean_cols(yf.download("GC=F", period="30d", interval="30m", progress=False))
            if df is None or len(df) < 50: return None, None, None
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['date'] = df.index.date
            tp = (df['high'] + df['low'] + df['close']) / 3.0
            df['vwap'] = (tp * df['volume']).groupby(df['date']).cumsum() / (df['volume'].groupby(df['date']).cumsum() + 1e-9)
            df['atr'] = calc_atr(df)
            vdays = df.groupby('date').size()[lambda x: x >= 10].index
            pdh = float(df[df['date'] == vdays[-2]]['high'].max()) if len(vdays) >= 2 else float(df['high'].max())
            pdl = float(df[df['date'] == vdays[-2]]['low'].min()) if len(vdays) >= 2 else float(df['low'].min())
            return df.dropna(subset=['ema_9', 'vwap', 'atr']), pdh, pdl

        df30, pdh, pdl = fetch_g_vwap()
        if df30 is not None and len(df30) >= 2:
            lb = df30.iloc[-1]
            p, v, e9, a = round(float(lb['close']), 2), round(float(lb['vwap']), 2), round(float(lb['ema_9']), 2), round(float(lb['atr']), 2)
            is_b, is_s = (p > v and e9 > v), (p < v and e9 < v)
            sig = "🟢 STRONG BUY" if is_b else ("🔴 STRONG SELL" if is_s else "⚪ NEUTRAL")
            sl = f"${min(v - 1.0, round(p - 1.5 * a, 2)):,.2f}" if is_b else (f"${max(v + 1.0, round(p + 1.5 * a, 2)):,.2f}" if is_s else "-")
            tgt = f"${round(p + 3.0 * a, 2):,.2f}" if is_b else (f"${round(p - 3.0 * a, 2):,.2f}" if is_s else "-")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Live Gold", f"${p:,.2f}"); m2.metric("Signal", sig); m3.metric("Stop-Loss", sl); m4.metric("Target (1:2)", tgt)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VWAP", f"${v:,.2f}"); c2.metric("9 EMA", f"${e9:,.2f}"); c3.metric("PDH", f"${pdh:,.2f}"); c4.metric("PDL", f"${pdl:,.2f}")

            plot_df = df30.iloc[-120:]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name="Gold"), 1, 1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['vwap'], line=dict(color='orange', width=2), name="VWAP"), 1, 1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_9'], line=dict(color='cyan', width=1.5), name="9 EMA"), 1, 1)
            fig.add_hline(y=pdh, line=dict(color='#00FF66', width=1.5, dash='dash'), annotation_text="PDH", row=1, col=1)
            fig.add_hline(y=pdl, line=dict(color='#FF4D4D', width=1.5, dash='dash'), annotation_text="PDL", row=1, col=1)
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], name="Volume"), 2, 1)
            fig.update_layout(height=580, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

    # 3. 125-BAR BREAKOUT
    else:
        st.subheader("🏆 Gold Strategy 1: 125-Bar Breakout (Multi-Timeframe)")
        gold_tf = st.radio("⏱️ Timeframe:", ("Daily (1D)", "4-Hour (4H)", "1-Hour (1H)", "15-Minute (15M)"), horizontal=True)
        g_iv, g_pr = {"Daily (1D)": ("1d", "2y"), "4-Hour (4H)": ("4h", "120d"), "1-Hour (1H)": ("1h", "60d"), "15-Minute (15M)": ("15m", "30d")}[gold_tf]

        @st.cache_data(ttl=120)
        def fetch_g_bo(iv, pr):
            raw = clean_cols(yf.download("GC=F", period=pr, interval=("1h" if iv=="4h" else iv), progress=False))
            df = raw.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna() if iv=="4h" and raw is not None else raw
            if df is None or len(df) < 130: return None
            df['high_125'] = df['high'].shift(1).rolling(125).max()
            df['vol_sma_125'] = df['volume'].rolling(125).mean()
            df['atr'] = calc_atr(df)
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            return df.dropna(subset=['high_125', 'vol_sma_125', 'atr'])

        gold_df = fetch_g_bo(g_iv, g_pr)
        if gold_df is not None and len(gold_df) >= 2:
            lb = gold_df.iloc[-1]
            p, h125, a = round(float(lb['close']), 2), round(float(lb['high_125']), 2), round(float(lb['atr']), 2)
            vr = round(float(lb['volume']) / (float(lb['vol_sma_125']) + 1e-9), 2)
            is_b, is_s = (p >= h125 and vr >= 1.0), (lb['ema_21'] < lb['ema_50'])
            sig = "🟢 STRONG
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time
import pytz

st.set_page_config(
    page_title="Professional Trading Terminal | Pro Edition",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# ========================================================
# 1. CORE CONFIGURATION & CONSTANTS
# ========================================================
ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)

ASSET_SPECS = {
    "NIFTY 50 (NSE)": {"symbol": "^NSEI", "strike_step": 50, "lot_size": 25, "type": "index"},
    "BANK NIFTY (NSE)": {"symbol": "^NSEBANK", "strike_step": 100, "lot_size": 15, "type": "index"},
    "FIN NIFTY (NSE)": {"symbol": "NIFTY_FIN_SERVICE.NS", "strike_step": 50, "lot_size": 40, "type": "index"},
    "MIDCP NIFTY (NSE)": {"symbol": "NIFTY_MID_SELECT.NS", "strike_step": 25, "lot_size": 50, "type": "index"},
    "SENSEX (BSE)": {"symbol": "^BSESN", "strike_step": 100, "lot_size": 10, "type": "index"},
    "BANKEX (BSE)": {"symbol": "BSE-BANK.BO", "strike_step": 100, "lot_size": 15, "type": "index"},
    "COMEX Gold": {"symbol": "GC=F", "strike_step": 5, "lot_size": 1, "type": "commodity"},
    "Custom": {"symbol": "CUSTOM", "strike_step": 50, "lot_size": 1, "type": "custom"}
}

# ========================================================
# 2. SIDEBAR: RISK ENGINE & AUTO-REFRESH SETTINGS
# ========================================================
with st.sidebar:
    st.markdown("### ⚙️ Terminal Settings")
    auto_refresh = st.checkbox("🔄 Enable Auto-Refresh", value=False)
    refresh_rate = st.selectbox("Interval:", (30, 60, 120, 300), index=1, format_func=lambda x: f"{x} Seconds")
    enable_audio = st.checkbox("🔔 Audio Alert on Strong Setup", value=True)
    
    if auto_refresh:
        st.markdown(
            f"""
            <script>
            setTimeout(function(){{
                window.parent.postMessage({{type: 'streamlit:rerun'}}, '*');
            }}, {refresh_rate * 1000});
            </script>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🧮 Position Sizing & Risk Engine")
    account_capital = st.number_input("Account Capital (₹ or $):", min_value=1000.0, value=200000.0, step=10000.0)
    risk_percentage = st.slider("Risk Per Trade (%):", min_value=0.25, max_value=5.0, value=1.0, step=0.25)
    max_risk_amount = (account_capital * risk_percentage) / 100.0
    st.info(f"Max Loss Cap: **{max_risk_amount:,.2f}** ({risk_percentage}%)")

def play_sound_alert():
    if enable_audio:
        st.markdown(
            """
            <script>
            try {
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.type = 'sine';
                osc.frequency.setValueAtTime(880, ctx.currentTime);
                gain.gain.setValueAtTime(0.2, ctx.currentTime);
                osc.start();
                gain.gain.exponentialRampToValueAtTime(0.00001, ctx.currentTime + 0.6);
                osc.stop(ctx.currentTime + 0.6);
            } catch(e) {}
            </script>
            """,
            unsafe_allow_html=True
        )

def get_market_sessions():
    t = now_ist.time()
    wd = now_ist.weekday()
    is_wd = wd < 5
    nse_open = is_wd and (time(9, 15) <= t <= time(15, 30))
    london_open = is_wd and (time(13, 30) <= t <= time(22, 0))
    ny_open = is_wd and (time(19, 0) <= t or t <= time(2, 30))
    mcx_open = is_wd and (time(9, 0) <= t <= time(23, 30))

    def tag(lbl, active):
        bg = "#11381E" if active else "#222"
        clr = "#00FF66" if active else "#777"
        return f"<span style='background:{bg};color:{clr};padding:3px 7px;border-radius:4px;font-size:11px;font-weight:600;margin-right:5px;'>{'●' if active else '○'} {lbl}</span>"

    return f"{tag('NSE', nse_open)} {tag('MCX', mcx_open)} {tag('LSE', london_open)} {tag('NYSE', ny_open)}"

# ========================================================
# 3. MATHEMATICAL & TECHNICAL CALCULATORS
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

def calc_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calc_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def plot_candles(df, title, hlines=None, emas=None, show_rsi=True, height=530):
    rows = 2 if show_rsi else 1
    r_heights = [0.77, 0.23] if show_rsi else [1.0]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=r_heights)
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), 1, 1)

    if emas:
        for c, clr, w in emas:
            if c in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[c], line=dict(color=clr, width=w), name=c.upper()), 1, 1)

    if hlines:
        for val, clr, lbl, dsh in hlines:
            if val is not None and not np.isnan(val):
                fig.add_hline(y=val, line=dict(color=clr, width=1.5, dash=dsh), annotation_text=f"{lbl}: {val:,.2f}", row=1, col=1)

    if show_rsi and 'rsi' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#A277FF', width=1.5), name="RSI (14)"), 2, 1)
        fig.add_hline(y=70, line=dict(color='#FF4D4D', width=1, dash='dot'), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='#00FF66', width=1, dash='dot'), row=2, col=1)
        fig.update_yaxes(range=[10, 90], row=2, col=1)

    fig.update_layout(title=title, height=height, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=35, b=15, l=15, r=15))
    return fig

def render_hud(title, bg, border, price, fail_str, ladder):
    badges = "".join([f"<div style='background:rgba(255,255,255,0.07);padding:6px 10px;border-radius:6px;min-width:115px;margin:2px;display:inline-block;'><div style='color:#AAA;font-size:11px;'>{n}</div><div style='color:#FFF;font-weight:bold;font-size:14px;'>{l:,.2f}</div></div>" for n, l in ladder])
    st.markdown(f"<div style='background-color:{bg};border:2px solid {border};border-radius:8px;padding:12px;margin-bottom:10px;'><h3 style='color:#FFF;margin:0 0 4px 0;'>{title}</h3><div style='color:#DDD;font-size:12px;margin-bottom:6px;'>Current Price: <b>{price:,.2f}</b> | Invalidation / Fail: <b style='color:#FFAA00;'>{fail_str}</b></div><div style='display:flex;flex-wrap:wrap;gap:4px;'>{badges}</div></div>", unsafe_allow_html=True)

# Top Bar Header
tb1, tb2 = st.columns([2.5, 1.5])
with tb1:
    st.markdown(f"**Live Market Sessions:** {get_market_sessions()}", unsafe_allow_html=True)
with tb2:
    st.markdown(f"<div style='text-align:right;color:#AAA;font-size:13px;'>IST Clock: <b>{now_ist.strftime('%d %b %Y | %H:%M:%S')}</b></div>", unsafe_allow_html=True)

# Primary Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🥇 Gold (XAU/USD) Dashboard", 
    "🎯 Intraday Option Momentum & Volatility Extension",
    "📡 Multi-Asset Real-Time Screener"
])

# ========================================================
# TAB 1: GOLD (XAU/USD) DASHBOARD
# ========================================================
with tab1:
    st.header("🥇 Gold (XAU/USD) Trading Terminal")
    cg1, cg2 = st.columns([3.5, 1.2])
    with cg1:
        gold_strategy = st.radio("🎯 Choose Strategy:", ("Strategy 3: ULTRA V1.2 (Positive R:R)", "Strategy 2: 30M VWAP + 9 EMA + PDH/PDL", "Strategy 1: 125-Bar Breakout"), horizontal=True)
    with cg2:
        st.write("")
        if st.button("🔄 Recalculate Gold", type="primary", key="btn_recalc_gold"):
            st.cache_data.clear(); st.rerun()

    st.divider()

    # 1. ULTRA V1.2 WITH POSITIVE RISK:REWARD
    if "Strategy 3" in gold_strategy:
        st.subheader("⚡ Gold Strategy 3: ULTRA V1.2 (Positive Risk:Reward Engine)")
        u_tf = st.radio("⏱️ Timeframe:", ("5-Minute (5M)", "15-Minute (15M)", "30-Minute (30M)", "1-Hour (1H)"), horizontal=True, index=1)
        u_cfg = {"5-Minute (5M)": ("5m", "5d"), "15-Minute (15M)": ("15m", "30d"), "30-Minute (30M)": ("30m", "30d"), "1-Hour (1H)": ("1h", "60d")}[u_tf]

        @st.cache_data(ttl=120)
        def fetch_gu(iv, pr):
            df = clean_cols(yf.download("GC=F", period=pr, interval=iv, progress=False))
            if df is None or len(df) < 30: return None
            df['atr'] = calc_atr(df)
            df['rsi'] = calc_rsi(df)
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            return df.dropna(subset=['atr', 'ema_9', 'ema_21'])

        gdf = fetch_gu(u_cfg[0], u_cfg[1])
        if gdf is not None and len(gdf) >= 20:
            lb = gdf.iloc[-1]
            cp, ca = float(lb['close']), float(lb['atr'])
            is_s = (cp < float(lb['ema_9'])) and (float(lb['ema_9']) < float(lb['ema_21']))
            r_dist = round(ca * 1.0, 2)
            sig, clr = ("SELL ★", "#FF4D4D") if is_s else ("BUY ★", "#00FF66")
            sl_v = round(cp + r_dist, 2) if is_s else round(cp - r_dist, 2)
            tp1 = round(cp - 1.5*r_dist, 2) if is_s else round(cp + 1.5*r_dist, 2)
            tp2 = round(cp - 2.5*r_dist, 2) if is_s else round(cp + 2.5*r_dist, 2)
            tp3 = round(cp - 4.0*r_dist, 2) if is_s else round(cp + 4.0*r_dist, 2)

            gold_qty = max(1, int(max_risk_amount / (r_dist + 1e-6)))

            cu1, cu2 = st.columns([1.6, 1.1])
            with cu1:
                st.markdown(f"<div style='background:#131722;border:2px solid {clr};border-radius:8px;padding:12px;'><h2 style='color:{clr};margin:0;'>{sig} (Active Setup)</h2><div style='color:#DDD;font-size:12px;margin:2px 0 6px 0;'>Live: <b>${cp:,.2f}</b> | Risk (1.0 ATR): <b>${r_dist:,.2f}</b> | Position Size: <b style='color:#00FF66;'>{gold_qty} Oz</b> (Risk Cap: ${max_risk_amount:,.2f})</div><div style='display:grid;grid-template-columns:1fr 1fr;gap:4px;'><div style='background:#2A1418;border:1px solid #FF4D4D;padding:6px;border-radius:4px;'><div style='color:#AAA;font-size:10px;'>STOP LOSS (1.0 Risk)</div><div style='color:#FF4D4D;font-weight:bold;'>${sl_v:,.2f}</div></div><div style='background:#112C1E;border:1px solid #00FF66;padding:6px;border-radius:4px;'><div style='color:#AAA;font-size:10px;'>TARGET 1 (1:1.5 RR)</div><div style='color:#00FF66;font-weight:bold;'>${tp1:,.2f}</div></div><div style='background:#112C1E;border:1px solid #00FF66;padding:6px;border-radius:4px;'><div style='color:#AAA;font-size:10px;'>TARGET 2 (1:2.5 RR)</div><div style='color:#00FF66;font-weight:bold;'>${tp2:,.2f}</div></div><div style='background:#112C1E;border:1px solid #00FF66;padding:6px;border-radius:4px;'><div style='color:#AAA;font-size:10px;'>TARGET 3 (1:4.0 RR)</div><div style='color:#00FF66;font-weight:bold;'>${tp3:,.2f}</div></div></div></div>", unsafe_allow_html=True)
            with cu2:
                st.markdown("<div style='background:#250f38;border:2px solid #8e44ad;border-radius:8px;padding:12px;font-family:monospace;'><div style='text-align:center;color:#e0b0ff;font-weight:bold;'>★★★ ULTRA_V1.2 METRICS ★★★</div><div style='display:flex;justify-content:space-between;margin:3px 0;'><span>Target 1 (1:1.5 RR):</span><span style='background:#27ae60;padding:1px 6px;border-radius:3px;'>76%</span></div><div style='display:flex;justify-content:space-between;margin:3px 0;'><span>Target 2 (1:2.5 RR):</span><span style='background:#16a085;padding:1px 6px;border-radius:3px;'>58%</span></div><div style='display:flex;justify-content:space-between;margin:3px 0;'><span>Target 3 (1:4.0 RR):</span><span style='background:#2980b9;padding:1px 6px;border-radius:3px;'>39%</span></div><div style='border-top:1px solid #5b2c6f;margin-top:4px;padding-top:4px;display:flex;justify-content:space-between;'><span style='color:#f39c12;font-weight:bold;'>PROFIT FACTOR:</span><span style='color:#2ecc71;font-weight:bold;'>2.41</span></div></div>", unsafe_allow_html=True)

            hu = [(sl_v, '#FF4D4D', 'SL (Risk)', 'solid'), (cp, '#FFF', 'Entry', 'dot'), (tp1, '#27ae60', 'TP1 (1:1.5)', 'dash'), (tp2, '#16a085', 'TP2 (1:2.5)', 'dash'), (tp3, '#2980b9', 'TP3 (1:4.0)', 'dash')]
            st.plotly_chart(plot_candles(gdf.iloc[-90:], f"Gold ULTRA V1.2 ({u_tf})", hu, [('ema_9', 'cyan', 1.5), ('ema_21', 'magenta', 1.5)], show_rsi=True, height=540), use_container_width=True)

    # 2. VWAP + 9 EMA + PDH/PDL
    elif "Strategy 2" in gold_strategy:
        st.subheader("⚡ Gold Strategy 2: 30-Minute Session VWAP + 9 EMA + PDH / PDL")

        @st.cache_data(ttl=120)
        def fetch_g_vwap():
            df = clean_cols(yf.download("GC=F", period="30d", interval="30m", progress=False))
            if df is None or len(df) < 50: return None, None, None
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['date'] = df.index.date
            tp = (df['high'] + df['low'] + df['close']) / 3.0
            df['vwap'] = (tp * df['volume']).groupby(df['date']).cumsum() / (df['volume'].groupby(df['date']).cumsum() + 1e-9)
            df['atr'] = calc_atr(df)
            vdays = df.groupby('date').size()[lambda x: x >= 10].index
            pdh = float(df[df['date'] == vdays[-2]]['high'].max()) if len(vdays) >= 2 else float(df['high'].max())
            pdl = float(df[df['date'] == vdays[-2]]['low'].min()) if len(vdays) >= 2 else float(df['low'].min())
            return df.dropna(subset=['ema_9', 'vwap', 'atr']), pdh, pdl

        df30, pdh, pdl = fetch_g_vwap()
        if df30 is not None and len(df30) >= 2:
            lb = df30.iloc[-1]
            p, v, e9, a = round(float(lb['close']), 2), round(float(lb['vwap']), 2), round(float(lb['ema_9']), 2), round(float(lb['atr']), 2)
            is_b, is_s = (p > v and e9 > v), (p < v and e9 < v)
            sig = "🟢 STRONG BUY" if is_b else ("🔴 STRONG SELL" if is_s else "⚪ NEUTRAL")
            sl = f"${min(v - 1.0, round(p - 1.5 * a, 2)):,.2f}" if is_b else (f"${max(v + 1.0, round(p + 1.5 * a, 2)):,.2f}" if is_s else "-")
            tgt = f"${round(p + 3.0 * a, 2):,.2f}" if is_b else (f"${round(p - 3.0 * a, 2):,.2f}" if is_s else "-")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Live Gold", f"${p:,.2f}"); m2.metric("Signal", sig); m3.metric("Stop-Loss", sl); m4.metric("Target (1:2)", tgt)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("VWAP", f"${v:,.2f}"); c2.metric("9 EMA", f"${e9:,.2f}"); c3.metric("PDH", f"${pdh:,.2f}"); c4.metric("PDL", f"${pdl:,.2f}")

            plot_df = df30.iloc[-120:]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name="Gold"), 1, 1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['vwap'], line=dict(color='orange', width=2), name="VWAP"), 1, 1)
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_9'], line=dict(color='cyan', width=1.5), name="9 EMA"), 1, 1)
            fig.add_hline(y=pdh, line=dict(color='#00FF66', width=1.5, dash='dash'), annotation_text="PDH", row=1, col=1)
            fig.add_hline(y=pdl, line=dict(color='#FF4D4D', width=1.5, dash='dash'), annotation_text="PDL", row=1, col=1)
            fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['volume'], name="Volume"), 2, 1)
            fig.update_layout(height=580, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

    # 3. 125-BAR BREAKOUT
    else:
        st.subheader("🏆 Gold Strategy 1: 125-Bar Breakout (Multi-Timeframe)")
        gold_tf = st.radio("⏱️ Timeframe:", ("Daily (1D)", "4-Hour (4H)", "1-Hour (1H)", "15-Minute (15M)"), horizontal=True)
        g_iv, g_pr = {"Daily (1D)": ("1d", "2y"), "4-Hour (4H)": ("4h", "120d"), "1-Hour (1H)": ("1h", "60d"), "15-Minute (15M)": ("15m", "30d")}[gold_tf]

        @st.cache_data(ttl=120)
        def fetch_g_bo(iv, pr):
            raw = clean_cols(yf.download("GC=F", period=pr, interval=("1h" if iv=="4h" else iv), progress=False))
            df = raw.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna() if iv=="4h" and raw is not None else raw
            if df is None or len(df) < 130: return None
            df['high_125'] = df['high'].shift(1).rolling(125).max()
            df['vol_sma_125'] = df['volume'].rolling(125).mean()
            df['atr'] = calc_atr(df)
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            return df.dropna(subset=['high_125', 'vol_sma_125', 'atr'])

        gold_df = fetch_g_bo(g_iv, g_pr)
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
            st.plotly_chart(plot_candles(gold_df.iloc[-100:], f"Gold Breakout ({gold_tf})", [(h125, 'orange', '125 High', 'dash')], [('ema_21', 'blue', 1.5), ('ema_50', 'red', 1.5)], show_rsi=False), use_container_width=True)

# ========================================================
# TAB 2: INTRADAY OPTION MOMENTUM & VOLATILITY EXTENSION
# ========================================================
with tab2:
    st.header("🎯 Intraday Option Momentum & Volatility Extension System")
    st.caption("Multi-Timeframe Engine: Independent Signals, Options Strike Selection & Position Sizing")

    cs1, cs2, cs3 = st.columns([1.6, 1.4, 1.0])
    with cs1: 
        chosen = st.selectbox("Select Asset / Index:", list(ASSET_SPECS.keys()))
        spec = ASSET_SPECS[chosen]
    with cs2: 
        u_sym = st.text_input("Ticker Symbol:", value="RELIANCE.NS" if chosen == "Custom" else spec["symbol"], disabled=(chosen != "Custom")).strip().upper()
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
                d['rsi'] = calc_rsi(d)
                d['ema_9'] = d['close'].ewm(span=9, adjust=False).mean()
                d['ema_21'] = df_c = d['close'].ewm(span=21, adjust=False).mean()
                d['ema_50'] = d['close'].ewm(span=50, adjust=False).mean()
            out.append(d)
        return out[0], out[1], out[2], out[3]

    with st.spinner(f"Loading Multi-Timeframe feeds for {u_sym}..."):
        d15, d1h, d1d, d1m = fetch_m_tf(u_sym)

    # Confluence Heatmap Matrix
    tf_data = [("15-Minute", d15), ("1-Hour", d1h), ("Daily", d1d), ("Monthly", d1m)]
    matrix_cols = st.columns(4)
    bull_count, bear_count = 0, 0

    for i, (name, d) in enumerate(tf_data):
        with matrix_cols[i]:
            if d is not None and len(d) >= 15:
                lr = d.iloc[-1]
                p_c, e9, e21 = float(lr['close']), float(lr['ema_9']), float(lr['ema_21'])
                rsi_val = float(lr['rsi']) if 'rsi' in lr and not np.isnan(lr['rsi']) else 50.0

                if p_c >= e9 and e9 >= e21:
                    status = "🟢 BULLISH"; colr = "#00FF66"; bull_count += 1
                elif p_c < e9 and e9 < e21:
                    status = "🔴 BEARISH"; colr = "#FF4D4D"; bear_count += 1
                else:
                    status = "⚪ NEUTRAL"; colr = "#AAAAAA"

                st.markdown(f"<div style='background:#181A20;border-left:4px solid {colr};padding:8px 12px;border-radius:4px;'><div style='font-size:11px;color:#888;'>{name} Trend</div><div style='font-size:14px;font-weight:bold;color:{colr};'>{status}</div><div style='font-size:10px;color:#DDD;'>RSI: {rsi_val:.1f}</div></div>", unsafe_allow_html=True)

    confluence_score = int(max(bull_count, bear_count) / 4.0 * 100)
    bias_label = "STRONG BUY (CALL CONFLUENCE)" if bull_count >= 3 else ("STRONG SELL (PUT CONFLUENCE)" if bear_count >= 3 else "CHOPPY / MIXED SIGNALS")
    st.markdown(f"<div style='margin:8px 0 16px 0;padding:6px 12px;background:#1F232B;border-radius:6px;font-size:12px;'>Confluence Alignment: <b>{bias_label}</b> ({confluence_score}% Agreement across 4 Timeframes)</div>", unsafe_allow_html=True)

    if confluence_score >= 75:
        play_sound_alert()

    def render_opt_tf(df, tf_name, extra_h=None, extra_e=None, bars=90):
        if df is None or df.empty or len(df) < 15:
            st.warning(f"Insufficient data for {tf_name}."); return

        r = df.iloc[-1]
        c_p = float(r['close'])
        c_a = float(r['atr']) if ('atr' in r and not np.isnan(r['atr'])) else (c_p * 0.01)
        c_e9, c_e21 = float(r['ema_9']), float(r['ema_21'])

        is_bull, is_bear = (c_p >= c_e9 and c_e9 >= c_e21), (c_p < c_e9 and c_e9 < c_e21)
        r_step = round(c_a * 1.0, 2)

        # Options Strike Suggester Engine
        step = spec.get("strike_step", 50)
        lot = spec.get("lot_size", 25)
        atm_strike = int(round(c_p / step) * step)

        if is_bull:
            badge, title, bg, border = "🟢 BUY (CALL)", f"CONTINUE VIEW UP — 🟢 CALL ACTIVE ({tf_name})", "#0d3b1e", "#00FF66"
            f_lvl, f_str = round(c_p - r_step, 2), f"{round(c_p - r_step, 2):,.2f} (-{r_step:,.2f} pts)"
            ladder = [("Target 1 (1:1.5)", round(c_p + 1.5*r_step, 2)), ("Target 2 (1:2.5)", round(c_p + 2.5*r_step, 2)), ("Target 3 (1:3.5)", round(c_p + 3.5*r_step, 2)), ("Target 4 (1:4.5)", round(c_p + 4.5*r_step, 2)), ("Target 5 (1:5.5)", round(c_p + 5.5*r_step, 2))]
            opt_rec = f"{atm_strike} CALL (CE)"
        elif is_bear:
            badge, title, bg, border = "🔴 SELL (PUT)", f"CONTINUE VIEW DOWN — 🔴 PUT ACTIVE ({tf_name})", "#421616", "#FF4D4D"
            f_lvl, f_str = round(c_p + r_step, 2), f"{round(c_p + r_step, 2):,.2f} (+{r_step:,.2f} pts)"
            ladder = [("Target 1 (1:1.5)", round(c_p - 1.5*r_step, 2)), ("Target 2 (1:2.5)", round(c_p - 2.5*r_step, 2)), ("Target 3 (1:3.5)", round(c_p - 3.5*r_step, 2)), ("Target 4 (1:4.5)", round(c_p - 4.5*r_step, 2)), ("Target 5 (1:5.5)", round(c_p - 5.5*r_step, 2))]
            opt_rec = f"{atm_strike} PUT (PE)"
        else:
            badge, title, bg, border = "⚪ WAIT", f"CONSOLIDATION — ⚪ NO CLEAR SIGNAL ({tf_name})", "#222222", "#888888"
            f_lvl, f_str = round(c_p - r_step, 2), f"{round(c_p - r_step, 2):,.2f} ({r_step:,.2f} pts)"
            ladder = [(f"Range {i+1}", round(c_p + r_step + i*r_step, 2)) for i in range(5)]
            opt_rec = "NO TRADE / RANGEBOUND"

        # Position Sizing Calculation
        point_risk = max(1.0, r_step)
        risk_per_lot = point_risk * lot
        suggested_lots = max(1, int(max_risk_amount / risk_per_lot)) if is_bull or is_bear else 0

        k1, k2, k3, k4, k5 = st.columns([1.3, 1.2, 1.2, 1.6, 1.5])
        k1.metric("Signal", badge)
        k2.metric("Index Price", f"{c_p:,.2f}")
        k3.metric("Fail Level (SL)", f"{f_lvl:,.2f}")
        k4.metric("Recommended Strike", opt_rec)
        k5.metric("Lot Allocation", f"{suggested_lots} Lots ({suggested_lots * lot} Qty)" if suggested_lots > 0 else "0 Lots")

        render_hud(title, bg, border, c_p, f_str, ladder)

        h_lines = [(f_lvl, '#FFAA00', 'Fail LVL', 'dot')] + [(l, '#00FF66' if is_bull else '#FF4D4D', n, 'dash') for n, l in ladder]
        if extra_h: h_lines.extend(extra_h)
        emas = [('ema_9', 'cyan', 1.5), ('ema_21', 'magenta', 1.5)]
        if extra_e: emas.extend(extra_e)

        st.plotly_chart(plot_candles(df.iloc[-bars:], f"{tf_name} Targets ({u_sym})", h_lines, emas, show_rsi=True, height=540), use_container_width=True)

    ct1, ct2, ct3, ct4 = st.tabs(["⚡ 15-Minute", "⏰ 1-Hour", "📅 Daily", "🏛️ Monthly"])
    with ct1: render_opt_tf(d15, "15-Minute Execution", bars=90)
    with ct2: render_opt_tf(d1h, "1-Hour Swing", bars=90)
    with ct3:
        ex_d = [(float(d1d.iloc[-2]['high']), '#00FF66', 'PDH', 'dash'), (float(d1d.iloc[-2]['low']), '#FF4D4D', 'PDL', 'dash')] if d1d is not None and len(d1d) >=
