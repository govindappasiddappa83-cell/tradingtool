import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pytz

st.set_page_config(page_title="Professional Trading Terminal", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")
ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)

ASSET_SPECS = {
    "NIFTY 50": {"symbol": "^NSEI", "step": 50, "lot": 25, "class": "Index"},
    "BANK NIFTY": {"symbol": "^NSEBANK", "step": 100, "lot": 15, "class": "Index"},
    "FIN NIFTY": {"symbol": "NIFTY_FIN_SERVICE.NS", "step": 50, "lot": 40, "class": "Index"},
    "RELIANCE": {"symbol": "RELIANCE.NS", "step": 20, "lot": 250, "class": "Stock"},
    "HDFC BANK": {"symbol": "HDFCBANK.NS", "step": 10, "lot": 550, "class": "Stock"},
    "SPOT GOLD": {"symbol": "XAUUSD=X", "step": 5, "lot": 1, "class": "Commodity"},
    "COMEX GOLD": {"symbol": "GC=F", "step": 5, "lot": 1, "class": "Commodity"},
    "USD / INR": {"symbol": "INR=X", "step": 0.25, "lot": 1000, "class": "Forex"},
    "EUR / USD": {"symbol": "EURUSD=X", "step": 0.005, "lot": 10000, "class": "Forex"}
}

TF_CONFIG = {
    "3M": {"interval": "1m", "period": "5d", "resample": "3min"},
    "15M": {"interval": "15m", "period": "30d", "resample": None},
    "1H": {"interval": "1h", "period": "60d", "resample": None},
    "1D": {"interval": "1d", "period": "1y", "resample": None}
}

with st.sidebar:
    st.markdown("### ⚙️ Terminal Settings")
    auto_refresh = st.checkbox("🔄 Enable Auto-Refresh", value=False)
    refresh_rate = st.selectbox("Interval (sec):", (15, 30, 60, 120), index=1)
    if auto_refresh:
        st.markdown(f"<script>setTimeout(function(){{ window.parent.postMessage({{type: 'streamlit:rerun'}}, '*'); }}, {refresh_rate * 1000});</script>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧮 Risk Management")
    account_capital = st.number_input("Capital (₹ / $):", min_value=1000.0, value=200000.0, step=10000.0)
    risk_pct = st.slider("Risk Per Trade (%):", 0.25, 5.0, 1.0, 0.25)
    max_risk_amount = (account_capital * risk_pct) / 100.0
    st.info(f"Max Loss Allowed: **{max_risk_amount:,.2f}**")

def clean_cols(df):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    req = ['open', 'high', 'low', 'close', 'volume']
    return df[req].dropna().copy() if all(c in df.columns for c in req) else None

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
    df['date'] = df.index.date
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    df['vwap'] = (tp * df['volume']).groupby(df['date']).cumsum() / (df['volume'].groupby(df['date']).cumsum() + 1e-9)
    return df

@st.cache_data(ttl=15)
def load_feed(symbol, tf_key):
    cfg = TF_CONFIG[tf_key]
    raw = yf.download(symbol, period=cfg["period"], interval=cfg["interval"], progress=False)
    df = clean_cols(raw)
    if (df is None or df.empty) and symbol in ["XAUUSD=X", "GC=F"]:
        alt = "GC=F" if symbol == "XAUUSD=X" else "XAUUSD=X"
        df = clean_cols(yf.download(alt, period=cfg["period"], interval=cfg["interval"], progress=False))
    if df is None or df.empty: return None
    if cfg["resample"]:
        df = df.resample(cfg["resample"]).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    return compute_indicators(df)

def calculate_setup(cp, atr, e9, e21, step, lot, asset_class):
    is_bull = cp >= e9 >= e21
    is_bear = cp < e9 < e21
    atm = int(round(cp / step) * step) if step >= 1 else round(cp, 4)
    if is_bull:
        sig = "BUY 🟢"
        sl = round(cp - atr, 4 if cp < 500 else 2)
        tp1 = round(cp + 1.5 * atr, 4 if cp < 500 else 2)
        tp2 = round(cp + 2.5 * atr, 4 if cp < 500 else 2)
        strike = f"{atm} CE" if asset_class in ["Index", "Stock"] else "LONG CALL"
        lots = max(1, int(max_risk_amount / (max(atr, 0.001) * lot)))
    elif is_bear:
        sig = "SELL 🔴"
        sl = round(cp + atr, 4 if cp < 500 else 2)
        tp1 = round(cp - 1.5 * atr, 4 if cp < 500 else 2)
        tp2 = round(cp - 2.5 * atr, 4 if cp < 500 else 2)
        strike = f"{atm} PE" if asset_class in ["Index", "Stock"] else "SHORT PUT"
        lots = max(1, int(max_risk_amount / (max(atr, 0.001) * lot)))
    else:
        sig = "WAIT ⚪"
        sl = round(cp - atr, 4 if cp < 500 else 2)
        tp1 = round(cp + atr, 4 if cp < 500 else 2)
        tp2 = round(cp + 2 * atr, 4 if cp < 500 else 2)
        strike = "NO TRADE"
        lots = 0
    return sig, sl, tp1, tp2, strike, lots

def draw_chart(df, title, lines=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_9'], line=dict(color='cyan', width=1.2), name="EMA 9"), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_21'], line=dict(color='magenta', width=1.2), name="EMA 21"), 1, 1)
    if 'vwap' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], line=dict(color='orange', width=1.2), name="VWAP"), 1, 1)
    if lines:
        for val, clr, txt in lines:
            if val is not None and not np.isnan(val):
                fmt = f"{val:,.4f}" if val < 500 else f"{val:,.2f}"
                fig.add_hline(y=val, line=dict(color=clr, width=1.3, dash='dash'), annotation_text=f"{txt}: {fmt}", row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#A277FF', width=1.3), name="RSI"), 2, 1)
    fig.add_hline(y=70, line=dict(color='#FF4D4D', dash='dot'), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='#00FF66', dash='dot'), row=2, col=1)
    fig.update_layout(title=title, height=450, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=30, b=10, l=10, r=10))
    return fig

# HEADER
hc1, hc2 = st.columns([3, 1.2])
hc1.markdown(f"**Live Market Clock (IST):** `{now_ist.strftime('%d %b %Y | %H:%M:%S')}`")
if hc2.button("🔄 Refresh Data & Clear Cache", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🥇 Gold (XAU/USD)", 
    "🎯 Intraday Option Momentum", 
    "📡 Multi-Asset Screener", 
    "🏛️ Goldman Sachs QIS Quant Hub",
    "🚀 Asymmetric Unlimited Profit Hedging"
])

# TAB 1: GOLD
with tab1:
    st.subheader("🥇 Gold Execution Hub")
    gc1, gc2 = st.columns([2, 1])
    g_feed = gc1.selectbox("Feed:", ["Spot Gold (XAUUSD=X)", "COMEX Gold (GC=F)"], index=0)
    g_tf = gc2.selectbox("Timeframe:", ["3M", "15M", "1H", "1D"], index=1, key="gtf")
    g_sym = "XAUUSD=X" if "Spot" in g_feed else "GC=F"
    gdf = load_feed(g_sym, g_tf)
    if gdf is not None and len(gdf) >= 15:
        r = gdf.iloc[-1]
        cp, atr = float(r['close']), float(r['atr'])
        sig, sl, tp1, tp2, strike, lots = calculate_setup(cp, atr, float(r['ema_9']), float(r['ema_21']), 5, 1, "Commodity")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Signal", sig)
        m2.metric("Live Gold Price", f"${cp:,.2f}")
        m3.metric("Stop Loss", f"${sl:,.2f}")
        m4.metric("Target 1", f"${tp1:,.2f}")
        m5.metric("Position Size", f"{lots} Oz")
        st.plotly_chart(draw_chart(gdf.iloc[-90:], f"Gold ({g_tf})", [(sl, '#FF4D4D', 'SL'), (tp1, '#00FF66', 'TP1'), (tp2, '#00E5FF', 'TP2')]), use_container_width=True)
    else: st.warning("Loading Gold data...")

# TAB 2: OPTION MOMENTUM
with tab2:
    st.subheader("🎯 Options Strike Selection & Confluence")
    chosen = st.selectbox("Select Asset:", list(ASSET_SPECS.keys()))
    spec = ASSET_SPECS[chosen]
    data_map = {tf: load_feed(spec["symbol"], tf) for tf in ["3M", "15M", "1H", "1D"]}
    
    hcols = st.columns(4)
    bull_c, bear_c = 0, 0
    for i, (tf, d) in enumerate(data_map.items()):
        with hcols[i]:
            if d is not None:
                lr = d.iloc[-1]
                p, e9, e21 = float(lr['close']), float(lr['ema_9']), float(lr['ema_21'])
                status, clr = ("BULLISH 🟢", "#00FF66") if p >= e9 >= e21 else (("BEARISH 🔴", "#FF4D4D") if p < e9 < e21 else ("NEUTRAL ⚪", "#888"))
                if "BULL" in status: bull_c += 1
                elif "BEAR" in status: bear_c += 1
                st.markdown(f"<div style='background:#181A20;border-left:4px solid {clr};padding:8px;'><small>{tf}</small><br><b>{status}</b></div>", unsafe_allow_html=True)
    st.markdown(f"**Confluence Alignment:** `{int(max(bull_c, bear_c)/4.0*100)}%` across all timeframes")
    
    sub_tabs = st.tabs(["⚡ 3M", "⚡ 15M", "⏰ 1H", "📅 1D"])
    for i, (tf, stab) in enumerate(zip(["3M", "15M", "1H", "1D"], sub_tabs)):
        with stab:
            df = data_map[tf]
            if df is not None:
                r = df.iloc[-1]
                cp, atr = float(r['close']), float(r['atr'])
                sig, sl, tp1, tp2, strike, lots = calculate_setup(cp, atr, float(r['ema_9']), float(r['ema_21']), spec["step"], spec["lot"], spec["class"])
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Signal", sig); c2.metric("Price", f"{cp:,.2f}"); c3.metric("Strike", strike); c4.metric("SL", f"{sl:,.2f}"); c5.metric("Lots", f"{lots}")
                st.plotly_chart(draw_chart(df.iloc[-90:], f"{chosen} ({tf})", [(sl, '#FFAA00', 'SL'), (tp1, '#00FF66', 'TP1'), (tp2, '#00E5FF', 'TP2')]), use_container_width=True)

# TAB 3: SCREENER WITH FULL ACTIONABLE LEVELS
with tab3:
    st.subheader("📡 Real-Time Multi-Asset Actionable Execution Screener")
    scanner_tf = st.selectbox("Select Timeframe:", ["3M", "15M", "1H", "1D"], index=1, key="sc_tf")
    rows, feed_cache = [], {}
    with st.spinner(f"Scanning all markets on {scanner_tf}..."):
        for name, meta in ASSET_SPECS.items():
            df = load_feed(meta["symbol"], scanner_tf)
            if df is not None and len(df) >= 15:
                feed_cache[name] = df
                r = df.iloc[-1]
                cp, atr = float(r['close']), float(r['atr'])
                sig, sl, tp1, tp2, strike, lots = calculate_setup(cp, atr, float(r['ema_9']), float(r['ema_21']), meta["step"], meta["lot"], meta["class"])
                rows.append({
                    "Asset": name,
                    "Action Signal": sig,
                    "Price": round(cp, 4 if cp < 500 else 2),
                    "Stop Loss (SL)": sl,
                    "Target 1 (TP1)": tp1,
                    "Target 2 (TP2)": tp2,
                    "Recommended Strike": strike,
                    "Lots": lots,
                    "RSI": round(float(r['rsi']), 1)
                })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.subheader("🔍 Deep Dive Inspection on Any Asset")
        sel_asset = st.selectbox("Select Asset to Inspect from Scanner:", list(feed_cache.keys()))
        adf = feed_cache[sel_asset]
        ameta = ASSET_SPECS[sel_asset]
        ar = adf.iloc[-1]
        acp, aatr = float(ar['close']), float(ar['atr'])
        asig, asl, atp1, atp2, astrike, alots = calculate_setup(acp, aatr, float(ar['ema_9']), float(ar['ema_21']), ameta["step"], ameta["lot"], ameta["class"])
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Signal", asig); d2.metric("Price", f"{acp:,.2f}"); d3.metric("SL", f"{asl:,.2f}"); d4.metric("TP1", f"{atp1:,.2f}"); d5.metric("Target Strike", astrike)
        st.plotly_chart(draw_chart(adf.iloc[-90:], f"{sel_asset} Execution Chart ({scanner_tf})", [(asl, '#FF4D4D', 'SL'), (atp1, '#00FF66', 'TP1'), (atp2, '#00E5FF', 'TP2')]), use_container_width=True)

# TAB 4: GOLDMAN SACHS QUANT HUB
with tab4:
    st.subheader("🏛️ Goldman Sachs QIS Factor Model")
    f1, f2 = st.columns([2, 1])
    gs_asset = f1.selectbox("Asset for Analysis:", list(ASSET_SPECS.keys()), index=0)
    gs_tf = f2.selectbox("Factor Timeframe:", ["3M", "15M", "1H", "1D"], index=2, key="gs_tf")
    gs_df = load_feed(ASSET_SPECS[gs_asset]["symbol"], gs_tf)
    if gs_df is not None and len(gs_df) >= 30:
        gs_df['log_ret'] = np.log(gs_df['close'] / gs_df['close'].shift(1))
        ann_mult = 252 * (78 if gs_tf == "3M" else 26 if gs_tf == "15M" else 7 if gs_tf == "1H" else 1)
        gs_df['realized_vol'] = gs_df['log_ret'].rolling(20).std() * np.sqrt(ann_mult)
        spread = gs_df['close'].rolling(20).mean() - gs_df['close'].rolling(50).mean()
        gs_df['z_score'] = spread / (spread.rolling(50).std() + 1e-9)
        last = gs_df.iloc[-1]
        p, z, vol = float(last['close']), float(last['z_score']), float(last['realized_vol'])
        q_sig = "STRONG BUY 🟢" if z >= 1.0 else ("STRONG SHORT 🔴" if z <= -1.0 else "EQUILIBRIUM ⚪")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Quant Signal", q_sig); m2.metric("Price", f"{p:,.2f}"); m3.metric("Z-Score", f"{z:.2f} σ"); m4.metric("Realized Vol", f"{vol*100:.1f}%")
        st.plotly_chart(draw_chart(gs_df.iloc[-90:], f"Goldman Sachs Engine: {gs_asset} ({gs_tf})"), use_container_width=True)

# TAB 5: HEDGING
with tab5:
    st.subheader("🚀 Asymmetric Unlimited Profit Hedging")
    h_asset = st.selectbox("Select Asset to Trade:", ["NIFTY 50", "BANK NIFTY", "RELIANCE", "HDFC BANK"])
    h_meta = ASSET_SPECS[h_asset]
    hdf = load_feed(h_meta["symbol"], "1D")
    if hdf is not None:
        hp = float(hdf.iloc[-1]['close'])
        step, lot = h_meta["step"], h_meta["lot"]
        atm = int(round(hp / step) * step)
        is_bull = float(hdf.iloc[-1]['ema_9']) >= float(hdf.iloc[-1]['ema_21'])
        short_k, long_k = (atm, atm + 2*step) if is_bull else (atm, atm - 2*step)
        opt_type = "CE" if is_bull else "PE"
        max_loss = (2 * step + 0.25 * step) * lot
        b1, b2, b3 = st.columns(3)
        b1.info(f"**Leg 1:** SELL 1x {short_k} {opt_type}")
        b2.success(f"**Leg 2:** BUY 2x {long_k} {opt_type}")
        b3.warning("**Upside:** 🚀 UNLIMITED")
        st.metric("Max Capped Loss", f"₹{max_loss:,.2f}")
        st.write("Daily Action: Exit if 60% max profit reached or roll legs on trend extension.")

# --- apy.py complete ---
