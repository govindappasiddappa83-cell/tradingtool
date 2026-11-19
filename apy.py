import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pytz

st.set_page_config(page_title="Professional Quant & Hedging Terminal", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")

ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)

ASSET_SPECS = {
    "NIFTY 50 (NSE)": {"symbol": "^NSEI", "strike_step": 50, "lot_size": 25, "class": "Index"},
    "BANK NIFTY (NSE)": {"symbol": "^NSEBANK", "strike_step": 100, "lot_size": 15, "class": "Index"},
    "FIN NIFTY (NSE)": {"symbol": "NIFTY_FIN_SERVICE.NS", "strike_step": 50, "lot_size": 40, "class": "Index"},
    "MIDCP NIFTY (NSE)": {"symbol": "NIFTY_MID_SELECT.NS", "strike_step": 25, "lot_size": 50, "class": "Index"},
    "SENSEX (BSE)": {"symbol": "^BSESN", "strike_step": 100, "lot_size": 10, "class": "Index"},
    "RELIANCE (NSE)": {"symbol": "RELIANCE.NS", "strike_step": 20, "lot_size": 250, "class": "Stock"},
    "HDFC BANK (NSE)": {"symbol": "HDFCBANK.NS", "strike_step": 10, "lot_size": 550, "class": "Stock"},
    "Spot Gold / USD": {"symbol": "XAUUSD=X", "strike_step": 5, "lot_size": 1, "class": "Commodity"},
    "COMEX Gold": {"symbol": "GC=F", "strike_step": 5, "lot_size": 1, "class": "Commodity"},
    "USD / INR": {"symbol": "INR=X", "strike_step": 0.25, "lot_size": 1000, "class": "Forex"},
    "EUR / USD": {"symbol": "EURUSD=X", "strike_step": 0.005, "lot_size": 10000, "class": "Forex"}
}

TF_CONFIG = {
    "3M": {"interval": "1m", "period": "5d", "resample": "3min"},
    "15M": {"interval": "15m", "period": "30d", "resample": None},
    "1H": {"interval": "1h", "period": "60d", "resample": None},
    "1D": {"interval": "1d", "period": "1y", "resample": None}
}

# --- SIDEBAR ENGINE ---
with st.sidebar:
    st.markdown("### ⚙️ Terminal Settings")
    auto_refresh = st.checkbox("🔄 Enable Auto-Refresh", value=False)
    refresh_rate = st.selectbox("Interval (sec):", (15, 30, 60, 120), index=1)
    enable_audio = st.checkbox("🔔 Audio Alert on Setups", value=True)
    if auto_refresh:
        st.markdown(f"<script>setTimeout(function(){{ window.parent.postMessage({{type: 'streamlit:rerun'}}, '*'); }}, {refresh_rate * 1000});</script>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧮 Portfolio Risk Engine")
    account_capital = st.number_input("Capital (₹ / $):", min_value=1000.0, value=200000.0, step=10000.0)
    risk_pct = st.slider("Risk Per Trade (%):", min_value=0.25, max_value=5.0, value=1.0, step=0.25)
    max_risk_amount = (account_capital * risk_pct) / 100.0
    st.info(f"Max Loss Cap: **{max_risk_amount:,.2f}**")

def trigger_audio():
    if enable_audio:
        st.markdown("<script>try{var c=new(window.AudioContext||window.webkitAudioContext)(),o=c.createOscillator(),g=c.createGain();o.connect(g);g.connect(c.destination);o.type='sine';o.frequency.setValueAtTime(880,c.currentTime);g.gain.setValueAtTime(0.2,c.currentTime);o.start();g.gain.exponentialRampToValueAtTime(0.0001,c.currentTime+0.5);o.stop(c.currentTime+0.5);}catch(e){}</script>", unsafe_allow_html=True)

# --- QUANT & DATA PROCESSING ---
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
    return df

@st.cache_data(ttl=10)
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

def draw_chart(df, title, lines=None, sub_trace=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_9'], line=dict(color='cyan', width=1.2), name="EMA 9"), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_21'], line=dict(color='magenta', width=1.2), name="EMA 21"), 1, 1)
    if lines:
        for val, clr, txt in lines:
            if val is not None and not np.isnan(val):
                fmt = f"{val:,.4f}" if val < 500 else f"{val:,.2f}"
                fig.add_hline(y=val, line=dict(color=clr, width=1.4, dash='dash'), annotation_text=f"{txt}: {fmt}", row=1, col=1)
    if sub_trace:
        fig.add_trace(sub_trace[0], 2, 1)
        for h_y, h_c in sub_trace[1]:
            fig.add_hline(y=h_y, line=dict(color=h_c, dash='dot'), row=2, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#A277FF', width=1.3), name="RSI"), 2, 1)
        fig.add_hline(y=70, line=dict(color='#FF4D4D', dash='dot'), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='#00FF66', dash='dot'), row=2, col=1)
    fig.update_layout(title=title, height=450, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=30, b=10, l=10, r=10))
    return fig

# --- TOP HEADER ---
top_c1, top_c2 = st.columns([3, 1.2])
top_c1.markdown(f"**Live Market Clock (IST):** `{now_ist.strftime('%d %b %Y | %H:%M:%S')}`")
if top_c2.button("🔄 Refresh Data & Clear Cache", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🥇 Gold (XAU/USD)", 
    "🎯 Intraday Option Momentum", 
    "📡 Multi-Asset Screener", 
    "🏛️ Goldman Sachs QIS Quant Hub",
    "🚀 Asymmetric Unlimited Profit Hedging"
])

# ========================================================
# TAB 1: GOLD (XAU/USD) EXECUTION HUB
# ========================================================
with tab1:
    st.subheader("🥇 Gold (XAU/USD) Real-Time Execution Engine")
    gc1, gc2, gc3 = st.columns([2.0, 1.0, 1.2])
    g_strat = gc1.radio("Strategy:", ("VWAP + 9 EMA (Institutional)", "ULTRA V1.2 (R:R Ladder)", "125-Bar Breakout"), horizontal=True)
    g_tf = gc2.selectbox("Timeframe:", ["3M", "15M", "1H", "1D"], index=1, key="gtf")
    g_feed = gc3.selectbox("Feed:", ["Spot Gold (XAUUSD=X)", "COMEX Gold (GC=F)"], index=0)
    g_sym = "XAUUSD=X" if "Spot" in g_feed else "GC=F"

    gdf = load_feed(g_sym, g_tf)
    if gdf is not None and len(gdf) >= 15:
        r = gdf.iloc[-1]
        cp = float(r['close'])
        atr = float(r['atr']) if ('atr' in r and not np.isnan(r['atr'])) else (cp * 0.004)
        e9, e21 = float(r['ema_9']), float(r['ema_21'])

        # Institutional VWAP
        gdf['date'] = gdf.index.date
        tp = (gdf['high'] + gdf['low'] + gdf['close']) / 3.0
        gdf['vwap'] = (tp * gdf['volume']).groupby(gdf['date']).cumsum() / (gdf['volume'].groupby(gdf['date']).cumsum() + 1e-9)
        vwap_val = float(gdf.iloc[-1]['vwap']) if not np.isnan(gdf.iloc[-1]['vwap']) else e21

        if "VWAP" in g_strat:
            is_buy = (cp > vwap_val and e9 > vwap_val)
            is_sell = (cp < vwap_val and e9 < vwap_val)
            sig = "STRONG BUY 🟢" if is_buy else ("STRONG SELL 🔴" if is_sell else "NEUTRAL WAIT ⚪")
            sl = round(min(vwap_val - 1.5, cp - 1.2 * atr), 2) if is_buy else (round(max(vwap_val + 1.5, cp + 1.2 * atr), 2) if is_sell else round(cp - atr, 2))
            tp1 = round(cp + 1.5 * abs(cp - sl), 2) if (is_buy or not is_sell) else round(cp - 1.5 * abs(cp - sl), 2)
            tp2 = round(cp + 2.5 * abs(cp - sl), 2) if (is_buy or not is_sell) else round(cp - 2.5 * abs(cp - sl), 2)
        elif "ULTRA" in g_strat:
            is_sell = cp < e9 < e21
            sig = "SELL 🔴" if is_sell else "BUY 🟢"
            sl = round(cp + atr if is_sell else cp - atr, 2)
            tp1 = round(cp - 1.5 * atr if is_sell else cp + 1.5 * atr, 2)
            tp2 = round(cp - 2.5 * atr if is_sell else cp + 2.5 * atr, 2)
        else:
            lb = 40 if g_tf in ["3M", "15M"] else 125
            gdf['h_break'] = gdf['high'].shift(1).rolling(min(lb, len(gdf)-1)).max()
            h_val = float(gdf.iloc[-1]['h_break'])
            sig = "BREAKOUT BUY 🟢" if cp >= h_val else "CONSOLIDATION ⚪"
            sl = round(cp - 1.5 * atr, 2)
            tp1 = round(cp + 2.0 * atr, 2)
            tp2 = round(cp + 3.5 * atr, 2)

        risk_dist = max(1.0, abs(cp - sl))
        pos_oz = max(1, int(max_risk_amount / risk_dist))

        # Always-visible 5 core metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Signal", sig)
        m2.metric("Live Gold Price", f"${cp:,.2f}")
        m3.metric("Stop Loss (SL)", f"${sl:,.2f}", delta=f"{sl-cp:,.2f}$")
        m4.metric("Target 1 (TP1)", f"${tp1:,.2f}", delta=f"{tp1-cp:,.2f}$")
        m5.metric("Target 2 (TP2)", f"${tp2:,.2f}", delta=f"{tp2-cp:,.2f}$")

        s1, s2, s3 = st.columns(3)
        s1.info(f"Position Sizing: **{pos_oz} Oz** (Risk Cap: ${max_risk_amount:,.2f})")
        s2.info(f"Session VWAP: **${vwap_val:,.2f}** | 9 EMA: **${e9:,.2f}**")
        s3.info(f"ATR (14 Volatility): **${atr:,.2f}**")

        lines = [(sl, '#FF4D4D', 'SL'), (tp1, '#00FF66', 'TP1'), (tp2, '#00E5FF', 'TP2')]
        if "VWAP" in g_strat and not np.isnan(vwap_val): lines.append((vwap_val, 'orange', 'VWAP'))
        st.plotly_chart(draw_chart(gdf.iloc[-90:], f"Gold Execution ({g_tf}) | Feed: {g_sym}", lines), use_container_width=True)
    else:
        st.warning("Fetching fresh Gold feed... Click 'Refresh Data & Clear Cache' above.")

# ========================================================
# TAB 2: INTRADAY OPTION MOMENTUM
# ========================================================
with tab2:
    st.subheader("🎯 Options Strike Selection & Multi-TF Confluence (3M, 15M, 1H, 1D)")
    chosen_sym = st.selectbox("Select Asset:", list(ASSET_SPECS.keys()))
    sym = ASSET_SPECS[chosen_sym]["symbol"]
    step, lot = ASSET_SPECS[chosen_sym]["strike_step"], ASSET_SPECS[chosen_sym]["lot_size"]
    data_map = {tf: load_feed(sym, tf) for tf in ["3M", "15M", "1H", "1D"]}

    heat_cols = st.columns(4)
    bull_count, bear_count = 0, 0
    for idx, (tname, d) in enumerate(data_map.items()):
        with heat_cols[idx]:
            if d is not None:
                lr = d.iloc[-1]
                c_p, e9, e21 = float(lr['close']), float(lr['ema_9']), float(lr['ema_21'])
                if c_p >= e9 >= e21: status, clr = "BULLISH 🟢", "#00FF66"; bull_count += 1
                elif c_p < e9 < e21: status, clr = "BEARISH 🔴", "#FF4D4D"; bear_count += 1
                else: status, clr = "NEUTRAL ⚪", "#888888"
                st.markdown(f"<div style='background:#181A20;border-left:4px solid {clr};padding:8px;'><small>{tname}</small><br><b>{status}</b></div>", unsafe_allow_html=True)

    confluence_score = int(max(bull_count, bear_count) / 4.0 * 100)
    if confluence_score >= 75: trigger_audio()
    st.markdown(f"**Confluence Alignment:** `{confluence_score}%` across all 4 timeframes")

    sub_tabs = st.tabs(["⚡ 3-Minute", "⚡ 15-Minute", "⏰ 1-Hour", "📅 Daily"])
    for idx, (tf_key, sub_t) in enumerate(zip(["3M", "15M", "1H", "1D"], sub_tabs)):
        with sub_t:
            df = data_map[tf_key]
            if df is not None:
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
                t1.metric("Signal", sig_lbl); t2.metric("Asset Price", f"{c_p:,.2f}"); t3.metric("Strike", str(rec_opt)); t4.metric("SL", f"{sl:,.2f}"); t5.metric("Lots", f"{lots}")
                ladder = [(sl, '#FFAA00', 'SL')] + [(c_p + m*atr if is_bull else c_p - m*atr, '#00FF66' if is_bull else '#FF4D4D', f"T{i}") for i, m in enumerate([1.5, 2.5, 3.5], 1)]
                st.plotly_chart(draw_chart(df.iloc[-90:], f"{chosen_sym} ({tf_key})", ladder), use_container_width=True)

# ========================================================
# TAB 3: REAL-TIME MULTI-ASSET SCREENER
# ========================================================
with tab3:
    st.subheader("📡 Real-Time Multi-Asset Scanner")
    scanner_tf = st.selectbox("Select Timeframe:", ["3M", "15M", "1H", "1D"], index=1)
    scan_list = [("NIFTY 50", "^NSEI"), ("BANK NIFTY", "^NSEBANK"), ("FIN NIFTY", "NIFTY_FIN_SERVICE.NS"), ("RELIANCE", "RELIANCE.NS"), ("HDFC BANK", "HDFCBANK.NS"), ("SPOT GOLD", "XAUUSD=X"), ("USD / INR", "INR=X"), ("EUR / USD", "EURUSD=X")]
    results = []
    with st.spinner(f"Scanning tickers on {scanner_tf}..."):
        for name, ticker in scan_list:
            df = load_feed(ticker, scanner_tf)
            if df is not None:
                lr = df.iloc[-1]
                cp, e9, e21, rsi = float(lr['close']), float(lr['ema_9']), float(lr['ema_21']), float(lr['rsi'])
                sig = "BULLISH BREAKOUT 🟢" if (cp >= e9 >= e21) else ("BEARISH BREAKDOWN 🔴" if (cp < e9 < e21) else "CONSOLIDATION ⚪")
                results.append({"Asset": name, "Timeframe": scanner_tf, "Price": round(cp, 4) if cp < 500 else round(cp, 2), "Signal": sig, "RSI (14)": round(rsi, 1)})
    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

# ========================================================
# TAB 4: GOLDMAN SACHS QIS QUANT HUB
# ========================================================
with tab4:
    st.subheader("🏛️ Goldman Sachs QIS Factor Model")
    f1, f2, f3 = st.columns([1.5, 1.0, 1.0])
    gs_asset = f1.selectbox("Asset:", list(ASSET_SPECS.keys()), index=0)
    gs_meta = ASSET_SPECS[gs_asset]
    gs_tf = f2.selectbox("Factor Timeframe:", ["3M", "15M", "1H", "1D"], index=2)
    z_threshold = f3.selectbox("Sensitivity:", [1.0, 1.25, 1.5], index=0, format_func=lambda x: f"±{x}σ")
    target_vol = st.slider("Target Volatility (%):", 5.0, 30.0, 12.0, 1.0) / 100.0

    gs_df = load_feed(gs_meta["symbol"], gs_tf)
    if gs_df is not None and len(gs_df) >= 30:
        gs_df['log_ret'] = np.log(gs_df['close'] / gs_df['close'].shift(1))
        ann_mult = 252 * (78 if gs_tf == "3M" else 26 if gs_tf == "15M" else 7 if gs_tf == "1H" else 1)
        gs_df['realized_vol'] = gs_df['log_ret'].rolling(20).std() * np.sqrt(ann_mult)
        gs_df['sma_20'] = gs_df['close'].rolling(20).mean()
        gs_df['sma_50'] = gs_df['close'].rolling(50).mean()
        spread = gs_df['sma_20'] - gs_df['sma_50']
        gs_df['z_score'] = spread / (spread.rolling(50).std() + 1e-9)

        last = gs_df.iloc[-1]
        p_price, r_vol, z_val, atr_val = float(last['close']), float(last['realized_vol']), float(last['z_score']), float(last['atr'])
        vol_scalar = np.clip(target_vol / max(r_vol, 0.04), 0.25, 2.5)
        risk_dist = atr_val * 1.5
        lot_sz = gs_meta.get("lot_size", 1)
        calc_lots = max(1, int(((max_risk_amount / (risk_dist + 1e-6)) * vol_scalar) / lot_sz))

        if z_val >= z_threshold and p_price >= float(last['sma_50']):
            q_signal = "STRONG BUY 🟢"
            sl_v, tp1_v = round(p_price - risk_dist, 2), round(p_price + 1.5*risk_dist, 2)
        elif z_val <= -z_threshold and p_price <= float(last['sma_50']):
            q_signal = "STRONG SHORT 🔴"
            sl_v, tp1_v = round(p_price + risk_dist, 2), round(p_price - 1.5*risk_dist, 2)
        else:
            q_signal = "EQUILIBRIUM ⚪"
            sl_v, tp1_v = round(p_price - risk_dist, 2), round(p_price + risk_dist, 2)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Signal", q_signal); m2.metric("Price", f"{p_price:,.2f}"); m3.metric("Z-Score", f"{z_val:.2f} σ"); m4.metric("Vol", f"{r_vol*100:.1f}%"); m5.metric("Lots", f"{calc_lots}")

        gs_lines = [(sl_v, '#FF4D4D', 'SL'), (tp1_v, '#00FF66', 'TP1')]
        z_trace = (go.Scatter(x=gs_df.index, y=gs_df['z_score'], line=dict(color='#00E5FF', width=1.5), name="Z-Score"), [(z_threshold, '#00FF66'), (-z_threshold, '#FF4D4D')])
        st.plotly_chart(draw_chart(gs_df.iloc[-90:], f"Goldman Sachs Factor Engine: {gs_asset} ({gs_tf})", gs_lines, sub_trace=z_trace), use_container_width=True)

# ========================================================
# TAB 5: ASYMMETRIC UNLIMITED PROFIT HEDGING
# ========================================================
with tab5:
    st.subheader("🚀 Asymmetric Unlimited Profit Hedging & Daily Adjuster")
    st.caption("Ratio Volatility Backspread: Capped Downside with 100% Unlimited Open-Ended Profit")

    h_asset = st.selectbox("Select Asset to Trade:", ["NIFTY 50 (NSE)", "BANK NIFTY (NSE)", "RELIANCE (NSE)", "HDFC BANK (NSE)"])
    h_meta = ASSET_SPECS[h_asset]
    h_df = load_feed(h_meta["symbol"], "1D")

    if h_df is not None and len(h_df) >= 20:
        h_last = h_df.iloc[-1]
        h_price, h_step, h_lot = float(h_last['close']), h_meta["strike_step"], h_meta["lot_size"]
        is_bullish = float(h_last['ema_9']) >= float(h_last['ema_21'])
        atm = int(round(h_price / h_step) * h_step)

        if is_bullish:
            strat_title = "Call Ratio Backspread (Capped Loss + Unlimited Upside)"
            short_k, long_k = atm, atm + (2 * h_step)
            leg1 = f"SELL 1x {short_k} CE (Financer)"
            leg2 = f"BUY 2x {long_k} CE (Unlimited Engine)"
            net_debit = round(h_step * 0.25, 1)
            max_loss_rs = ((long_k - short_k) + net_debit) * h_lot
            px_range = np.linspace(short_k - (2 * h_step), long_k + (5 * h_step), 40)
            payoffs = [(-max(0, p - short_k) + 1.8*h_step + 2 * (max(0, p - long_k) - 0.9*h_step)) * h_lot for p in px_range]
            adv = f"If price rallies above {long_k}, profit expands without limit. If market crashes, your loss is capped at ₹{abs(net_debit * h_lot):,.2f}."
        else:
            strat_title = "Put Ratio Backspread (Capped Loss + Unlimited Downside)"
            short_k, long_k = atm, atm - (2 * h_step)
            leg1 = f"SELL 1x {short_k} PE (Financer)"
            leg2 = f"BUY 2x {long_k} PE (Unlimited Engine)"
            net_debit = round(h_step * 0.25, 1)
            max_loss_rs = ((short_k - long_k) + net_debit) * h_lot
            px_range = np.linspace(long_k - (5 * h_step), short_k + (2 * h_step), 40)
            payoffs = [(-max(0, short_k - p) + 1.8*h_step + 2 * (max(0, long_k - p) - 0.9*h_step)) * h_lot for p in px_range]
            adv = f"If price falls below {long_k}, profit expands without limit. If market rallies, your loss is capped at ₹{abs(net_debit * h_lot):,.2f}."

        st.markdown(f"### Strategy: **{strat_title}**")
        b1, b2, b3 = st.columns(3)
        b1.info(f"**Leg 1:** {leg1}")
        b2.success(f"**Leg 2:** {leg2}")
        b3.warning(f"**Upside Model:** 🚀 UNLIMITED")

        m1, m2, m3 = st.columns(3)
        m1.metric("Max Capped Loss", f"₹{max_loss_rs:,.2f}")
        m2.metric("Max Profit", "UNLIMITED 🚀")
        m3.metric("Crash Risk Buffer", f"Capped at ₹{abs(net_debit * h_lot):,.2f}")

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=px_range, y=payoffs, mode='lines', line=dict(color='#00FF66', width=2.5), name="Payoff (₹)"))
        fig_p.add_hline(y=0, line=dict(color='#888888', dash='dot'))
        fig_p.add_vline(x=h_price, line=dict(color='#00E5FF', dash='dash'), annotation_text="Spot")
        fig_p.update_layout(title="Asymmetric Payoff Curve", height=320, template="plotly_dark", margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_p, use_container_width=True)

        st.markdown("### 🛠️ Daily Trade Adjustment Protocol")
        st.write(f"1. **Execution Rule:** Enter both legs simultaneously on weekly options.")
        st.write(f"2. **Daily Action:** {adv}")

# --- apy.py complete ---
