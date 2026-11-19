import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time
import pytz
import requests

st.set_page_config(
    page_title="Professional Quant Terminal | Pro Edition",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# ========================================================
# 1. CORE CONFIGURATION & TIMEFRAME SPECS
# ========================================================
ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)

ASSET_SPECS = {
    # Indices
    "NIFTY 50 (NSE)": {"symbol": "^NSEI", "strike_step": 50, "lot_size": 25, "class": "Index"},
    "BANK NIFTY (NSE)": {"symbol": "^NSEBANK", "strike_step": 100, "lot_size": 15, "class": "Index"},
    "FIN NIFTY (NSE)": {"symbol": "NIFTY_FIN_SERVICE.NS", "strike_step": 50, "lot_size": 40, "class": "Index"},
    "MIDCP NIFTY (NSE)": {"symbol": "NIFTY_MID_SELECT.NS", "strike_step": 25, "lot_size": 50, "class": "Index"},
    "SENSEX (BSE)": {"symbol": "^BSESN", "strike_step": 100, "lot_size": 10, "class": "Index"},
    "S&P 500 (US)": {"symbol": "^GSPC", "strike_step": 10, "lot_size": 1, "class": "Index"},

    # Commodities (Gold)
    "COMEX Gold (GC=F)": {"symbol": "GC=F", "strike_step": 5, "lot_size": 1, "class": "Commodity"},
    "Spot Gold / USD (XAU/USD)": {"symbol": "XAUUSD=X", "strike_step": 5, "lot_size": 1, "class": "Commodity"},

    # Currencies / Forex
    "USD / INR (NSE FX)": {"symbol": "INR=X", "strike_step": 0.25, "lot_size": 1000, "class": "Forex"},
    "EUR / USD": {"symbol": "EURUSD=X", "strike_step": 0.005, "lot_size": 10000, "class": "Forex"},
    "GBP / USD": {"symbol": "GBPUSD=X", "strike_step": 0.005, "lot_size": 10000, "class": "Forex"},
    "USD / JPY": {"symbol": "JPY=X", "strike_step": 0.5, "lot_size": 10000, "class": "Forex"},
    "US Dollar Index (DXY)": {"symbol": "DX-Y.NYB", "strike_step": 0.5, "lot_size": 1, "class": "Macro Index"},

    "Custom Asset": {"symbol": "CUSTOM", "strike_step": 50, "lot_size": 1, "class": "Custom"}
}

TF_CONFIG = {
    "3M": {"interval": "1m", "period": "5d", "resample": "3min"},
    "15M": {"interval": "15m", "period": "30d", "resample": None},
    "1H": {"interval": "1h", "period": "60d", "resample": None},
    "1D": {"interval": "1d", "period": "1y", "resample": None}
}

# ========================================================
# 2. SIDEBAR: RISK ENGINE, TELEGRAM & AUTO-REFRESH
# ========================================================
with st.sidebar:
    st.markdown("### ⚙️ Terminal Settings")
    auto_refresh = st.checkbox("🔄 Enable Auto-Refresh", value=False)
    refresh_rate = st.selectbox("Interval (sec):", (30, 60, 120, 300), index=1)
    enable_audio = st.checkbox("🔔 Audio Alert on Setups", value=True)

    if auto_refresh:
        st.markdown(f"<script>setTimeout(function(){{ window.parent.postMessage({{type: 'streamlit:rerun'}}, '*'); }}, {refresh_rate * 1000});</script>", unsafe_allow_html=True)
        st.caption(f"⚡ Live polling every {refresh_rate}s active.")

    st.markdown("---")
    st.markdown("### 🧮 Portfolio Risk Engine")
    account_capital = st.number_input("Capital (₹ / $):", min_value=1000.0, value=200000.0, step=10000.0)
    risk_pct = st.slider("Risk Per Trade (%):", min_value=0.25, max_value=5.0, value=1.0, step=0.25)
    max_risk_amount = (account_capital * risk_pct) / 100.0
    st.info(f"Max Loss Allowed: **{max_risk_amount:,.2f}**")

    st.markdown("---")
    st.markdown("### 📱 Telegram Signal Bot")
    tg_token = st.text_input("Bot Token:", type="password", placeholder="Paste Bot Token here")
    tg_chat_id = st.text_input("Chat ID:", placeholder="Paste Chat ID here")

def send_telegram(msg):
    if tg_token and tg_chat_id:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": tg_chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=8)
            return True
        except Exception:
            return False
    return False

def trigger_audio():
    if enable_audio:
        st.markdown("""<script>
        try {
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = ctx.createOscillator(); var gain = ctx.createGain();
            osc.connect(gain); gain.connect(ctx.destination);
            osc.type = 'sine'; osc.frequency.setValueAtTime(880, ctx.currentTime);
            gain.gain.setValueAtTime(0.2, ctx.currentTime);
            osc.start(); gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.5);
            osc.stop(ctx.currentTime + 0.5);
        } catch(e) {}
        </script>""", unsafe_allow_html=True)

# ========================================================
# 3. DATA ENGINE & MATHEMATICAL HELPERS
# ========================================================
def clean_cols(df):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        l0 = [str(c).lower() for c in df.columns.get_level_values(0)]
        df.columns = df.columns.get_level_values(0 if any(k in l0 for k in ['close', 'open', 'high', 'low']) else 1)
    df.columns = [str(c).lower() for c in df.columns]
    req = ['open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in req): return None
    return df[req].dropna().copy()

def compute_indicators(df):
    if df is None or len(df) < 15: return None
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    return df

@st.cache_data(ttl=60)
def load_feed(symbol, tf_key):
    cfg = TF_CONFIG[tf_key]
    raw = yf.download(symbol, period=cfg["period"], interval=cfg["interval"], progress=False)
    df = clean_cols(raw)
    if df is None or df.empty: return None

    if cfg["resample"]:
        df = df.resample(cfg["resample"]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    return compute_indicators(df)

def draw_chart(df, title, lines=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.77, 0.23])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_9'], line=dict(color='cyan', width=1.2), name="EMA 9"), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_21'], line=dict(color='magenta', width=1.2), name="EMA 21"), 1, 1)
    if lines:
        for val, clr, txt in lines:
            if val and not np.isnan(val):
                fig.add_hline(y=val, line=dict(color=clr, width=1.3, dash='dash'), annotation_text=f"{txt}: {val:,.4f}" if val < 500 else f"{txt}: {val:,.2f}", row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#A277FF', width=1.3), name="RSI(14)"), 2, 1)
    fig.add_hline(y=70, line=dict(color='#FF4D4D', dash='dot'), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='#00FF66', dash='dot'), row=2, col=1)
    fig.update_layout(title=title, height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=30, b=10, l=10, r=10))
    return fig

# ========================================================
# 4. TOP EXECUTIVE HEADER WITH SCANNER REFRESH
# ========================================================
top_col1, top_col2 = st.columns([3, 1.2])
with top_col1:
    st.markdown(f"**Live Market Clock (IST):** `{now_ist.strftime('%d %b %Y | %H:%M:%S')}`")
with top_col2:
    if st.button("🔄 Refresh Data & Clear Cache", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

tab1, tab2, tab3, tab4 = st.tabs([
    "🥇 Gold (XAU/USD)", 
    "🎯 Intraday Option Momentum", 
    "📡 Multi-Asset Screener", 
    "🏛️ Goldman Sachs QIS Quant Hub (Gold & Currencies)"
])

# ========================================================
# TAB 1: GOLD DASHBOARD (3M, 15M, 1H, 1D)
# ========================================================
with tab1:
    st.subheader("🥇 Gold (XAU/USD) Execution Hub")
    g_col1, g_col2 = st.columns([2.5, 1.5])
    with g_col1:
        g_strat = st.radio("Gold Strategy:", ("ULTRA V1.2 (R:R Ladder)", "VWAP + 9 EMA", "125-Bar Breakout"), horizontal=True)
    with g_col2:
        g_tf = st.selectbox("Execution Timeframe:", ["3M", "15M", "1H", "1D"], index=1, key="gold_tf_select")

    gdf = load_feed("GC=F", g_tf)

    if gdf is not None and len(gdf) >= 15:
        r = gdf.iloc[-1]
        cp, atr = float(r['close']), float(r['atr'])

        if g_strat == "ULTRA V1.2 (R:R Ladder)":
            is_sell = cp < float(r['ema_9']) < float(r['ema_21'])
            sig, clr = ("SELL", "#FF4D4D") if is_sell else ("BUY", "#00FF66")
            sl = cp + atr if is_sell else cp - atr
            tp1 = cp - 1.5*atr if is_sell else cp + 1.5*atr
            tp2 = cp - 2.5*atr if is_sell else cp + 2.5*atr
            tp3 = cp - 4.0*atr if is_sell else cp + 4.0*atr
            pos_oz = max(1, int(max_risk_amount / (atr + 1e-6)))

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(f"Signal ({g_tf})", sig)
            c2.metric("Live Gold", f"${cp:,.2f}")
            c3.metric("Stop Loss", f"${sl:,.2f}")
            c4.metric("TP1 (1:1.5)", f"${tp1:,.2f}")
            c5.metric("Position Size", f"{pos_oz} Oz")

            st.plotly_chart(draw_chart(gdf.iloc[-90:], f"Gold ULTRA V1.2 ({g_tf})", [(sl, '#FF4D4D', 'SL'), (tp1, '#00FF66', 'TP1'), (tp2, '#00E5FF', 'TP2'), (tp3, '#B388FF', 'TP3')]), use_container_width=True)

        elif g_strat == "VWAP + 9 EMA":
            gdf['date'] = gdf.index.date
            tp = (gdf['high'] + gdf['low'] + gdf['close']) / 3.0
            gdf['vwap'] = (tp * gdf['volume']).groupby(gdf['date']).cumsum() / (gdf['volume'].groupby(gdf['date']).cumsum() + 1e-9)
            vwap, e9 = float(gdf.iloc[-1]['vwap']), float(gdf.iloc[-1]['ema_9'])
            sig = "BUY" if (cp > vwap and e9 > vwap) else ("SELL" if (cp < vwap and e9 < vwap) else "WAIT")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"Signal ({g_tf})", sig)
            c2.metric("Live Gold", f"${cp:,.2f}")
            c3.metric("VWAP", f"${vwap:,.2f}")
            c4.metric("9 EMA", f"${e9:,.2f}")

            st.plotly_chart(draw_chart(gdf.iloc[-90:], f"Gold VWAP + 9 EMA ({g_tf})", [(vwap, 'orange', 'VWAP')]), use_container_width=True)

        else:
            lookback = 60 if g_tf in ["3M", "15M"] else 125
            gdf['h_break'] = gdf['high'].shift(1).rolling(lookback).max()
            h_val = float(gdf.iloc[-1]['h_break'])
            sig = "BREAKOUT BUY" if cp >= h_val else "RANGE"

            c1, c2, c3 = st.columns(3)
            c1.metric(f"Signal ({g_tf})", sig)
            c2.metric("Live Gold", f"${cp:,.2f}")
            c3.metric(f"Breakout Level ({lookback} Bar)", f"${h_val:,.2f}")

            st.plotly_chart(draw_chart(gdf.iloc[-90:], f"Gold Breakout ({g_tf})", [(h_val, 'orange', 'High BO')]), use_container_width=True)
    else:
        st.warning(f"Data currently unavailable for Gold on {g_tf}.")

# ========================================================
# TAB 2: INTRADAY OPTION MOMENTUM (3M, 15M, 1H, 1D)
# ========================================================
with tab2:
    st.subheader("🎯 Options Strike Selection & Multi-TF Confluence (3M, 15M, 1H, 1D)")
    colA, colB = st.columns([2, 1])
    with colA: chosen_sym = st.selectbox("Select Target Index / Asset:", list(ASSET_SPECS.keys()))
    with colB: custom_t = st.text_input("Custom Symbol (if chosen):", value="RELIANCE.NS")

    sym = custom_t if chosen_sym == "Custom Asset" else ASSET_SPECS[chosen_sym]["symbol"]
    step = ASSET_SPECS[chosen_sym]["strike_step"]
    lot = ASSET_SPECS[chosen_sym]["lot_size"]

    # Preload all 4 standardized timeframes
    data_map = {
        "3M": load_feed(sym, "3M"),
        "15M": load_feed(sym, "15M"),
        "1H": load_feed(sym, "1H"),
        "1D": load_feed(sym, "1D")
    }

    # Confluence Heatmap
    heat_cols = st.columns(4)
    bull_count, bear_count = 0, 0
    for idx, (tname, d) in enumerate(data_map.items()):
        with heat_cols[idx]:
            if d is not None:
                lr = d.iloc[-1]
                c_p, e9, e21 = float(lr['close']), float(lr['ema_9']), float(lr['ema_21'])
                if c_p >= e9 >= e21:
                    status, clr = "BULLISH 🟢", "#00FF66"; bull_count += 1
                elif c_p < e9 < e21:
                    status, clr = "BEARISH 🔴", "#FF4D4D"; bear_count += 1
                else:
                    status, clr = "NEUTRAL ⚪", "#888888"
                st.markdown(f"<div style='background:#181A20;border-left:4px solid {clr};padding:8px;'><small>{tname} Alignment</small><br><b>{status}</b></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:#181A20;padding:8px;'><small>{tname}</small><br><b>NO DATA</b></div>", unsafe_allow_html=True)

    confluence_score = int(max(bull_count, bear_count) / 4.0 * 100)
    if confluence_score >= 75: trigger_audio()
    st.markdown(f"**Confluence Alignment:** `{confluence_score}%` across all 4 timeframes (3M, 15M, 1H, 1D)")

    sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs(["⚡ 3-Minute", "⚡ 15-Minute", "⏰ 1-Hour", "📅 Daily"])

    def display_option_engine(df, name):
        if df is None: st.warning(f"Data not available for {name}"); return
        r = df.iloc[-1]
        c_p, atr = float(r['close']), float(r['atr'])
        is_bull = float(r['close']) >= float(r['ema_9']) >= float(r['ema_21'])
        is_bear = float(r['close']) < float(r['ema_9']) < float(r['ema_21'])

        atm_strike = int(round(c_p / step) * step) if step >= 1 else round(c_p, 4)
        rec_opt = f"{atm_strike} CE" if is_bull else (f"{atm_strike} PE" if is_bear else "NO TRADE")
        sig_lbl = "CALL BUY 🟢" if is_bull else ("PUT BUY 🔴" if is_bear else "NEUTRAL ⚪")

        sl = c_p - atr if is_bull else c_p + atr
        lots = max(1, int(max_risk_amount / (atr * lot))) if (is_bull or is_bear) else 0

        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Signal", sig_lbl)
        t2.metric("Index/Asset Level", f"{c_p:,.4f}" if c_p < 500 else f"{c_p:,.2f}")
        t3.metric("Recommended Strike", str(rec_opt))
        t4.metric("SL Level", f"{sl:,.4f}" if sl < 500 else f"{sl:,.2f}")
        t5.metric("Lot Sizing", f"{lots} Lots ({lots*lot} Units)")

        ladder = [(sl, '#FFAA00', 'SL')]
        for i, mult in enumerate([1.5, 2.5, 3.5, 4.5, 5.5], 1):
            tgt = c_p + mult*atr if is_bull else c_p - mult*atr
            ladder.append((tgt, '#00FF66' if is_bull else '#FF4D4D', f"T{i}"))

        st.plotly_chart(draw_chart(df.iloc[-90:], f"{chosen_sym} - {name}", ladder), use_container_width=True)

    with sub_t1: display_option_engine(data_map["3M"], "3-Minute Scalp")
    with sub_t2: display_option_engine(data_map["15M"], "15-Minute Execution")
    with sub_t3: display_option_engine(data_map["1H"], "1-Hour Swing")
    with sub_t4: display_option_engine(data_map["1D"], "Daily Trend")

# ========================================================
# TAB 3: REAL-TIME MULTI-ASSET SCREENER (TIMEFRAME SELECTOR)
# ========================================================
with tab3:
    st.subheader("📡 Real-Time Multi-Asset Scanner (With Timeframe Engine)")
    
    s_col1, s_col2, s_col3 = st.columns([1.5, 1.5, 3])
    with s_col1:
        scanner_tf = st.selectbox("Select Scanner Timeframe:", ["3M", "15M", "1H", "1D"], index=1, key="scanner_tf_choice")
    with s_col2:
        st.write("")
        run_scan = st.button("🔎 Run Live Scanner", type="primary", use_container_width=True)
    with s_col3:
        st.caption(f"Currently scanning markets on the **{scanner_tf}** timeframe.")

    scan_list = [
        ("NIFTY 50", "^NSEI"),
        ("BANK NIFTY", "^NSEBANK"),
        ("FIN NIFTY", "NIFTY_FIN_SERVICE.NS"),
        ("COMEX GOLD", "GC=F"),
        ("USD / INR", "INR=X"),
        ("EUR / USD", "EURUSD=X"),
        ("GBP / USD", "GBPUSD=X"),
        ("US DOLLAR INDEX", "DX-Y.NYB")
    ]

    results = []
    with st.spinner(f"Scanning market tickers across {scanner_tf}..."):
        for name, ticker in scan_list:
            df = load_feed(ticker, scanner_tf)
            if df is not None:
                lr = df.iloc[-1]
                cp, e9, e21, rsi = float(lr['close']), float(lr['ema_9']), float(lr['ema_21']), float(lr['rsi'])
                if cp >= e9 >= e21: sig = "BULLISH BREAKOUT 🟢"
                elif cp < e9 < e21: sig = "BEARISH BREAKDOWN 🔴"
                else: sig = "CONSOLIDATION ⚪"
                results.append({
                    "Asset": name,
                    "Timeframe": scanner_tf,
                    "Last Price": round(cp, 4) if cp < 500 else round(cp, 2),
                    "Signal": sig,
                    "RSI (14)": round(rsi, 1)
                })

    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

# ========================================================
# TAB 4: GOLDMAN SACHS QIS QUANT HUB (3M, 15M, 1H, 1D)
# ========================================================
with tab4:
    st.subheader("🏛️ Goldman Sachs Quantitative Investment Strategies (QIS) Hub")
    st.caption("Institutional Multi-Asset Factor Model: Commodities (Gold), Forex Currencies & Global Indices")

    f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 1.2])
    with f_col1:
        cat_filter = st.radio("Asset Class Filter:", ("All Assets", "Commodities (Gold)", "Currencies / Forex", "Indices"), horizontal=True)
    with f_col2:
        def filter_assets(c):
            if c == "Commodities (Gold)": return [k for k, v in ASSET_SPECS.items() if v.get("class") == "Commodity"]
            if c == "Currencies / Forex": return [k for k, v in ASSET_SPECS.items() if v.get("class") in ["Forex", "Macro Index"]]
            if c == "Indices": return [k for k, v in ASSET_SPECS.items() if v.get("class") == "Index"]
            return list(ASSET_SPECS.keys())
        gs_asset = st.selectbox("Select Asset for Deep Quant Analysis:", filter_assets(cat_filter), index=0)
        gs_meta = ASSET_SPECS[gs_asset]
    with f_col3:
        gs_tf = st.selectbox("Quant Factor Timeframe:", ["3M", "15M", "1H", "1D"], index=2, key="gs_tf_choice")

    target_vol = st.slider("Target Portfolio Volatility (%):", min_value=5.0, max_value=30.0, value=12.0, step=1.0) / 100.0

    gs_df = load_feed(gs_meta["symbol"], gs_tf)

    if gs_df is not None and len(gs_df) >= 30:
        gs_df['log_ret'] = np.log(gs_df['close'] / gs_df['close'].shift(1))
        ann_factor = 252 * (78 if gs_tf == "3M" else 26 if gs_tf == "15M" else 7 if gs_tf == "1H" else 1)
        gs_df['realized_vol'] = gs_df['log_ret'].rolling(20).std() * np.sqrt(ann_factor)
        gs_df['sma_20'] = gs_df['close'].rolling(20).mean()
        gs_df['sma_50'] =
