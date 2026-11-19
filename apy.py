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
    page_title="Professional Quant Terminal | GS Macro Edition",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# ========================================================
# 1. CORE ASSET & MACRO CURRENCY REGISTRY
# ========================================================
ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)

ASSET_SPECS = {
    # Institutional Indices
    "NIFTY 50 (NSE)": {"symbol": "^NSEI", "strike_step": 50, "lot_size": 25, "class": "Index"},
    "BANK NIFTY (NSE)": {"symbol": "^NSEBANK", "strike_step": 100, "lot_size": 15, "class": "Index"},
    "FIN NIFTY (NSE)": {"symbol": "NIFTY_FIN_SERVICE.NS", "strike_step": 50, "lot_size": 40, "class": "Index"},
    "SENSEX (BSE)": {"symbol": "^BSESN", "strike_step": 100, "lot_size": 10, "class": "Index"},
    "S&P 500 (US)": {"symbol": "^GSPC", "strike_step": 10, "lot_size": 1, "class": "Index"},

    # Gold & Commodities
    "COMEX Gold (GC=F)": {"symbol": "GC=F", "strike_step": 5, "lot_size": 1, "class": "Commodity"},
    "Spot Gold / USD (XAU/USD)": {"symbol": "XAUUSD=X", "strike_step": 5, "lot_size": 1, "class": "Commodity"},

    # Institutional FX / Currencies
    "USD / INR (NSE FX)": {"symbol": "INR=X", "strike_step": 0.25, "lot_size": 1000, "class": "Forex"},
    "EUR / USD": {"symbol": "EURUSD=X", "strike_step": 0.005, "lot_size": 10000, "class": "Forex"},
    "GBP / USD": {"symbol": "GBPUSD=X", "strike_step": 0.005, "lot_size": 10000, "class": "Forex"},
    "USD / JPY": {"symbol": "JPY=X", "strike_step": 0.5, "lot_size": 10000, "class": "Forex"},
    "US Dollar Index (DXY)": {"symbol": "DX-Y.NYB", "strike_step": 0.5, "lot_size": 1, "class": "Macro Index"},

    "Custom Asset": {"symbol": "CUSTOM", "strike_step": 50, "lot_size": 1, "class": "Custom"}
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
# 3. MATHEMATICAL & TECHNICAL HELPERS
# ========================================================
def clean_data(df):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        l0 = [str(c).lower() for c in df.columns.get_level_values(0)]
        df.columns = df.columns.get_level_values(0 if any(k in l0 for k in ['close', 'open', 'high', 'low']) else 1)
    df.columns = [str(c).lower() for c in df.columns]
    req = ['open', 'high', 'low', 'close', 'volume']
    if not all(c in df.columns for c in req): return None
    df = df[req].dropna().copy()
    if len(df) < 15: return None
    
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

def draw_chart(df, title, lines=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.77, 0.23])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_9'], line=dict(color='cyan', width=1.2), name="EMA 9"), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_21'], line=dict(color='magenta', width=1.2), name="EMA 21"), 1, 1)
    if lines:
        for val, clr, txt in lines:
            if val and not np.isnan(val):
                fig.add_hline(y=val, line=dict(color=clr, width=1.3, dash='dash'), annotation_text=f"{txt}: {val:,.4f}", row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#A277FF', width=1.3), name="RSI(14)"), 2, 1)
    fig.add_hline(y=70, line=dict(color='#FF4D4D', dash='dot'), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='#00FF66', dash='dot'), row=2, col=1)
    fig.update_layout(title=title, height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=30, b=10, l=10, r=10))
    return fig

# --- TOP STATUS ---
st.markdown(f"**Live Market Clock (IST):** `{now_ist.strftime('%d %b %Y | %H:%M:%S')}`")
tab1, tab2, tab3, tab4 = st.tabs([
    "🥇 Gold (XAU/USD)", 
    "🎯 Intraday Option Momentum", 
    "📡 Multi-Asset Screener", 
    "🏛️ Goldman Sachs QIS Quant Hub (Gold & Currencies)"
])

# ========================================================
# TAB 1: GOLD DASHBOARD
# ========================================================
with tab1:
    st.subheader("🥇 Gold (XAU/USD) Execution Hub")
    g_strat = st.radio("Gold Strategy:", ("ULTRA V1.2 (R:R Ladder)", "30M VWAP + 9 EMA", "125-Bar Breakout"), horizontal=True)

    if g_strat == "ULTRA V1.2 (R:R Ladder)":
        gdf = clean_data(yf.download("GC=F", period="30d", interval="15m", progress=False))
        if gdf is not None:
            r = gdf.iloc[-1]
            cp, atr = float(r['close']), float(r['atr'])
            is_sell = cp < float(r['ema_9']) < float(r['ema_21'])
            sig, clr = ("SELL", "#FF4D4D") if is_sell else ("BUY", "#00FF66")
            sl = cp + atr if is_sell else cp - atr
            tp1 = cp - 1.5*atr if is_sell else cp + 1.5*atr
            tp2 = cp - 2.5*atr if is_sell else cp + 2.5*atr
            tp3 = cp - 4.0*atr if is_sell else cp + 4.0*atr
            pos_oz = max(1, int(max_risk_amount / (atr + 1e-6)))

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Signal", sig)
            c2.metric("Live Gold", f"${cp:,.2f}")
            c3.metric("Stop Loss", f"${sl:,.2f}")
            c4.metric("TP1 (1:1.5)", f"${tp1:,.2f}")
            c5.metric("Position Size", f"{pos_oz} Oz")

            st.plotly_chart(draw_chart(gdf.iloc[-90:], "Gold ULTRA V1.2 (15M)", [(sl, '#FF4D4D', 'SL'), (tp1, '#00FF66', 'TP1'), (tp2, '#00E5FF', 'TP2'), (tp3, '#B388FF', 'TP3')]), use_container_width=True)

    elif g_strat == "30M VWAP + 9 EMA":
        df30 = clean_data(yf.download("GC=F", period="30d", interval="30m", progress=False))
        if df30 is not None:
            df30['date'] = df30.index.date
            tp = (df30['high'] + df30['low'] + df30['close']) / 3.0
            df30['vwap'] = (tp * df30['volume']).groupby(df30['date']).cumsum() / (df30['volume'].groupby(df30['date']).cumsum() + 1e-9)
            r = df30.iloc[-1]
            cp, vwap, e9, atr = float(r['close']), float(r['vwap']), float(r['ema_9']), float(r['atr'])
            sig = "BUY" if (cp > vwap and e9 > vwap) else ("SELL" if (cp < vwap and e9 < vwap) else "WAIT")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signal", sig)
            c2.metric("Live Gold", f"${cp:,.2f}")
            c3.metric("VWAP", f"${vwap:,.2f}")
            c4.metric("9 EMA", f"${e9:,.2f}")

            st.plotly_chart(draw_chart(df30.iloc[-90:], "Gold 30M VWAP + 9 EMA", [(vwap, 'orange', 'VWAP')]), use_container_width=True)

    else:
        gbo = clean_data(yf.download("GC=F", period="60d", interval="1h", progress=False))
        if gbo is not None:
            gbo['h125'] = gbo['high'].shift(1).rolling(125).max()
            r = gbo.iloc[-1]
            cp, h125, atr = float(r['close']), float(r['h125']), float(r['atr'])
            sig = "BREAKOUT BUY" if cp >= h125 else "RANGE"

            c1, c2, c3 = st.columns(3)
            c1.metric("Signal", sig)
            c2.metric("Live Gold", f"${cp:,.2f}")
            c3.metric("125 High Level", f"${h125:,.2f}")

            st.plotly_chart(draw_chart(gbo.iloc[-90:], "Gold 125-Bar Breakout (1H)", [(h125, 'orange', '125 High')]), use_container_width=True)

# ========================================================
# TAB 2: INTRADAY OPTION MOMENTUM
# ========================================================
with tab2:
    st.subheader("🎯 Options Strike Selection & Multi-TF Ladder")
    colA, colB = st.columns([2, 1])
    with colA: chosen_sym = st.selectbox("Select Target Index / Asset:", list(ASSET_SPECS.keys()))
    with colB: custom_t = st.text_input("Custom Symbol (if chosen):", value="RELIANCE.NS")

    sym = custom_t if chosen_sym == "Custom Asset" else ASSET_SPECS[chosen_sym]["symbol"]
    step = ASSET_SPECS[chosen_sym]["strike_step"]
    lot = ASSET_SPECS[chosen_sym]["lot_size"]

    @st.cache_data(ttl=60)
    def fetch_tf(s):
        return {
            "15M": clean_data(yf.download(s, period="30d", interval="15m", progress=False)),
            "1H": clean_data(yf.download(s, period="60d", interval="1h", progress=False)),
            "1D": clean_data(yf.download(s, period="2y", interval="1d", progress=False)),
            "1M": clean_data(yf.download(s, period="10y", interval="1mo", progress=False))
        }

    data_map = fetch_tf(sym)

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
                st.markdown(f"<div style='background:#181A20;border-left:4px solid {clr};padding:8px;'><small>{tname} Bias</small><br><b>{status}</b></div>", unsafe_allow_html=True)

    confluence_score = int(max(bull_count, bear_count) / 4.0 * 100)
    if confluence_score >= 75: trigger_audio()
    st.markdown(f"**Confluence Alignment:** `{confluence_score}%` across all 4 timeframes")

    sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs(["⚡ 15-Minute", "⏰ 1-Hour", "📅 Daily", "🏛️ Monthly"])

    def display_option_engine(df, name):
        if df is None: st.warning("Data not available"); return
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

    with sub_t1: display_option_engine(data_map["15M"], "15-Minute Execution")
    with sub_t2: display_option_engine(data_map["1H"], "1-Hour Swing")
    with sub_t3: display_option_engine(data_map["1D"], "Daily Trend")
    with sub_t4: display_option_engine(data_map["1M"], "Monthly Structure")

# ========================================================
# TAB 3: REAL-TIME MULTI-ASSET SCREENER
# ========================================================
with tab3:
    st.subheader("📡 Real-Time Market Breakout Scanner")
    if st.button("🔎 Run Full Scanner", type="primary"):
        st.cache_data.clear()

    scan_list = [
        ("NIFTY 50", "^NSEI"),
        ("BANK NIFTY", "^NSEBANK"),
        ("COMEX GOLD", "GC=F"),
        ("USD / INR", "INR=X"),
        ("EUR / USD", "EURUSD=X"),
        ("GBP / USD", "GBPUSD=X"),
        ("US DOLLAR INDEX", "DX-Y.NYB")
    ]

    results = []
    with st.spinner("Scanning all multi-asset tickers..."):
        for name, ticker in scan_list:
            df = clean_data(yf.download(ticker, period="10d", interval="15m", progress=False))
            if df is not None:
                lr = df.iloc[-1]
                cp, e9, e21, rsi = float(lr['close']), float(lr['ema_9']), float(lr['ema_21']), float(lr['rsi'])
                if cp >= e9 >= e21: sig = "BULLISH BREAKOUT 🟢"
                elif cp < e9 < e21: sig = "BEARISH BREAKDOWN 🔴"
                else: sig = "CONSOLIDATION ⚪"
                results.append({"Asset": name, "Ticker": ticker, "Last Price": round(cp, 4) if cp < 500 else round(cp, 2), "Signal": sig, "RSI (14)": round(rsi, 1)})

    if results:
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

# ========================================================
# TAB 4: GOLDMAN SACHS QIS QUANT HUB (GOLD & CURRENCIES)
# ========================================================
with tab4:
    st.subheader("🏛️ Goldman Sachs Quantitative Investment Strategies (QIS) Hub")
    st.caption("Institutional Multi-Asset Factor Model: Commodities (Gold), Forex Currencies & Global Indices")

    # Filter Category selector
    cat_filter = st.radio("Asset Class Filter:", ("All Assets", "Commodities (Gold)", "Currencies / Forex", "Indices"), horizontal=True)

    def filter_assets(c):
        if c == "Commodities (Gold)": return [k for k, v in ASSET_SPECS.items() if v.get("class") == "Commodity"]
        if c == "Currencies / Forex": return [k for k, v in ASSET_SPECS.items() if v.get("class") in ["Forex", "Macro Index"]]
        if c == "Indices": return [k for k, v in ASSET_SPECS.items() if v.get("class") == "Index"]
        return list(ASSET_SPECS.keys())

    selectable_assets = filter_assets(cat_filter)

    q_col1, q_col2 = st.columns([2, 1])
    with q_col1:
        gs_asset = st.selectbox("Select Asset for Deep Quant Analysis:", selectable_assets, index=0)
        gs_meta = ASSET_SPECS[gs_asset]
    with q_col2:
        target_vol = st.slider("Target Portfolio Volatility (%):", min_value=5.0, max_value=30.0, value=12.0, step=1.0) / 100.0

    @st.cache_data(ttl=60)
    def compute_gs_quant(sym):
        df = clean_data(yf.download(sym, period="90d", interval="1h", progress=False))
        if df is None or len(df) < 55: return None

        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol'] = df['log_ret'].rolling(20).std() * np.sqrt(252)
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        spread = df['sma_20'] - df['sma_50']
        df['z_score'] = spread / (spread.rolling(50).std() + 1e-9)
        return df

    gs_df = compute_gs_quant(gs_meta["symbol"])

    if gs_df is not None:
        last = gs_df.iloc[-1]
        p_price = float(last['close'])
        r_vol = float(last['realized_vol'])
        z_val = float(last['z_score'])
        atr_val = float(last['atr'])

        # Institutional Inverse Volatility Sizing
        vol_scalar = np.clip(target_vol / max(r_vol, 0.04), 0.25, 2.5)
        risk_dist = atr_val * 1.5
        lot_sz = gs_meta.get("lot_size", 1)
        calc_lots = max(1, int(((max_risk_amount / (risk_dist + 1e-6)) * vol_scalar) / lot_sz))

        if z_val > 1.25 and p_price > float(last['sma_50']):
            q_signal, q_clr = "STRONG BUY 🟢", "#00FF66"
            sl_v = round(p_price - risk_dist, 4) if p_price < 500 else round(p_price - risk_dist, 2)
            tp1_v = round(p_price + 1.5*risk_dist, 4) if p_price < 500 else round(p_price + 1.5*risk_dist, 2)
            tp2_v = round(p_price + 2.5*risk_dist, 4) if p_price < 500 else round(p_price + 2.5*risk_dist, 2)
        elif z_val < -1.25 and p_price < float(last['sma_50']):
            q_signal, q_clr = "STRONG SHORT 🔴", "#FF4D4D"
            sl_v = round(p_price + risk_dist, 4) if p_price < 500 else round(p_price + risk_dist, 2)
            tp1_v = round(p_price - 1.5*risk_dist, 4) if p_price < 500 else round(p_price - 1.5*risk_dist, 2)
            tp2_v = round(p_price - 2.5*risk_dist, 4) if p_price < 500 else round(p_price - 2.5*risk_dist, 2)
        else:
            q_signal, q_clr = "EQUILIBRIUM / NEUTRAL ⚪", "#888888"
            sl_v = round(p_price - risk_dist, 4) if p_price < 500 else round(p_price - risk_dist, 2)
            tp1_v = round(p_price + risk_dist, 4) if p_price < 500 else round(p_price + risk_dist, 2)
            tp2_v = round(p_price + 2*risk_dist, 4) if p_price < 500 else round(p_price + 2*risk_dist, 2)

        # Executive Metrics
        p_str = f"{p_price:,.4f}" if p_price < 500 else f"${p_price:,.2f}" if "Gold" in gs_asset else f"{p_price:,.2f}"
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Quant Factor Signal", q_signal)
        m2.metric("Asset Price", p_str)
        m3.metric("Momentum Z-Score", f"{z_val:.2f} σ")
        m4.metric("Realized Volatility", f"{r_vol*100:.1f}%")
        m5.metric("Target Lot Allocation", f"{calc_lots} Lots ({calc_lots * lot_sz} Units)")

        # Telegram Instant Dispatcher
        st.markdown("---")
        t_col1, t_col2 = st.columns([3, 1])
        with t_col1:
            st.info(f"**Quant Setup ({gs_meta.get('class')}):** Entry: `{p_str}` | Stop-Loss: `{sl_v}` | Target 1: `{tp1_v}` | Target 2: `{tp2_v}`")
        with t_col2:
            if st.button("📲 Send Signal to Telegram", type="primary"):
                tg_msg = (
                    f"⚡ *GOLDMAN SACHS QIS QUANT ALERT*\n"
                    f"Asset: *{gs_asset}* (`{gs_meta['symbol']}`)\n"
                    f"Class: *{gs_meta.get('class')}*\n"
                    f"Signal: *{q_signal}*\n"
                    f"Entry Price: `{p_str}`\n"
                    f"Stop-Loss: `{sl_v}`\n"
                    f"Target 1: `{tp1_v}`\n"
                    f"Target 2: `{tp2_v}`\n"
                    f"Realized Vol: `{r_vol*100:.1f}%` (Scaler: {vol_scalar:.2f}x)\n"
                    f"Quant Allocation: `{calc_lots} Lots` ({calc_lots * lot_sz} Units)"
                )
                success = send_telegram(tg_msg)
                if success:
                    st.success("Alert successfully sent to your Telegram!")
                else:
                    st.error("Telegram error: Please verify your Bot Token & Chat ID in the sidebar.")

        # Quant Factor Charts
        fig_gs = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.7, 0.3])
        fig_gs.add_trace(go.Candlestick(x=gs_df.index, open=gs_df['open'], high=gs_df['high'], low=gs_df['low'], close=gs_df['close'], name="Price"), 1, 1)
        fig_gs.add_trace(go.Scatter(x=gs_df.index, y=gs_df['sma_20'], line=dict(color='yellow', width=1.2), name="Fast SMA (20)"), 1, 1)
        fig_gs.add_trace(go.Scatter(x=gs_df.index, y=gs_df['sma_50'], line=dict(color='cyan', width=1.5), name="Trend SMA (50)"), 1, 1)
        fig_gs.add_hline(y=sl_v, line=dict(color='#FF4D4D', dash='dash'), annotation_text="Quant SL", row=1, col=1)
        fig_gs.add_hline(y=tp1_v, line=dict(color='#00FF66', dash='dash'), annotation_text="Target 1", row=1, col=1)

        # Z-Score Subplot
        fig_gs.add_trace(go.Scatter(x=gs_df.index, y=gs_df['z_score'], line=dict(color='#00E5FF', width=1.5), name="Momentum Z-Score"), 2, 1)
        fig_gs.add_hline(y=1.25, line=dict(color='#00FF66', dash='dot'), row=2, col=1)
        fig_gs.add_hline(y=-1.25, line=dict(color='#FF4D4D', dash='dot'), row=2, col=1)
        fig_gs.update_layout(title=f"Goldman Sachs Factor Model: {gs_asset}", height=550, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_gs, use_container_width=True)
    else:
        st.warning(f"Insufficient historical data available for {gs_asset}.")
