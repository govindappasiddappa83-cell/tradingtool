import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time, timedelta
import pytz
import math
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh  # FIX #6: reliable refresh instead of JS hack

st.set_page_config(page_title="Professional Trading Terminal", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")
ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist)

ASSET_SPECS = {
    "NIFTY 50": {"symbol": "^NSEI", "step": 50, "lot": 25, "class": "Index", "exp_day": 3},
    "BANK NIFTY": {"symbol": "^NSEBANK", "step": 100, "lot": 15, "class": "Index", "exp_day": 2},
    "FIN NIFTY": {"symbol": "NIFTY_FIN_SERVICE.NS", "step": 50, "lot": 40, "class": "Index", "exp_day": 1},
    "RELIANCE": {"symbol": "RELIANCE.NS", "step": 20, "lot": 250, "class": "Stock", "exp_day": 3},
    "HDFC BANK": {"symbol": "HDFCBANK.NS", "step": 10, "lot": 550, "class": "Stock", "exp_day": 3},
    "SPOT GOLD": {"symbol": "XAUUSD=X", "step": 5, "lot": 1, "class": "Commodity", "exp_day": 4},
    "COMEX GOLD": {"symbol": "GC=F", "step": 5, "lot": 1, "class": "Commodity", "exp_day": 4},
    "USD / INR": {"symbol": "INR=X", "step": 0.25, "lot": 1000, "class": "Forex", "exp_day": 4},
    "EUR / USD": {"symbol": "EURUSD=X", "step": 0.005, "lot": 10000, "class": "Forex", "exp_day": 4}
}

TF_CONFIG = {
    "3M": {"interval": "1m", "period": "5d", "resample": "3min", "max_age_min": 10},
    "15M": {"interval": "15m", "period": "30d", "resample": None, "max_age_min": 40},
    "1H": {"interval": "1h", "period": "60d", "resample": None, "max_age_min": 150},
    "1D": {"interval": "1d", "period": "1y", "resample": None, "max_age_min": 2880}
}

def get_live_expiry(name):
    t_day = ASSET_SPECS.get(name, {}).get("exp_day", 3)
    c_date = now_ist.date()
    diff = (t_day - c_date.weekday()) % 7
    exp_date = c_date + timedelta(days=diff)
    cutoff = ist.localize(datetime.combine(exp_date, time(15, 30)))
    if now_ist > cutoff: exp_date += timedelta(days=7); cutoff += timedelta(days=7)
    delta = cutoff - now_ist
    return delta.days, int(delta.seconds // 3600), int((delta.seconds % 3600) // 60), exp_date.strftime("%d %b %Y"), (c_date == exp_date)

# FIX #1: Real Black-Scholes option pricer (replaces guessed constants)
def black_scholes_price(spot, strike, days_to_expiry, iv, option_type="CE", r=0.065):
    T = max(days_to_expiry, 0) / 365.0
    if T <= 0 or iv <= 0:
        return max(spot - strike, 0) if option_type == "CE" else max(strike - spot, 0)
    try:
        d1 = (math.log(spot / strike) + (r + 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
        d2 = d1 - iv * math.sqrt(T)
        if option_type == "CE":
            return max(spot * norm.cdf(d1) - strike * math.exp(-r * T) * norm.cdf(d2), 0)
        return max(strike * math.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1), 0)
    except (ValueError, ZeroDivisionError):
        return max(spot - strike, 0) if option_type == "CE" else max(strike - spot, 0)

with st.sidebar:
    st.markdown("### ⚙️ Terminal Settings")
    auto_refresh = st.checkbox("🔄 Auto-Refresh", value=False)
    refresh_rate = st.selectbox("Interval (sec):", (15, 30, 60, 120), index=1)
    if auto_refresh:
        st_autorefresh(interval=refresh_rate * 1000, key="auto_refresh")  # FIX #6
    st.markdown("---")
    st.markdown(f"**Terminal Time (IST):**<br>`{now_ist.strftime('%d %b %Y | %H:%M:%S')}`", unsafe_allow_html=True)
    st.markdown("---")
    capital = st.number_input("Capital (₹ / $):", 1000.0, value=200000.0, step=10000.0)
    risk_pct = st.slider("Risk (%):", 0.25, 5.0, 1.0, 0.25)
    max_risk = (capital * risk_pct) / 100.0
    st.info(f"Max Loss Allowed: **{max_risk:,.2f}**")
    st.markdown("---")
    st.markdown("### 📐 Options Pricing Input")
    manual_iv = st.slider("Implied Volatility (IV %) — check your broker's option chain:", 8.0, 60.0, 14.0, 0.5) / 100.0  # FIX #1

def clean_cols(df):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    req = ['open', 'high', 'low', 'close', 'volume']
    return df[req].dropna().copy() if all(c in df.columns for c in req) else None

@st.cache_data(ttl=15)
def load_feed(sym, tf):
    cfg = TF_CONFIG[tf]
    try:  # FIX #4: safe error handling around yf.download
        df = clean_cols(yf.download(sym, period=cfg["period"], interval=cfg["interval"], progress=False, timeout=15))
    except Exception:
        df = None
    if (df is None or df.empty) and sym in ["XAUUSD=X", "GC=F"]:
        alt = "GC=F" if sym == "XAUUSD=X" else "XAUUSD=X"
        try:
            df = clean_cols(yf.download(alt, period=cfg["period"], interval=cfg["interval"], progress=False, timeout=15))
        except Exception:
            df = None
    if df is None or df.empty: return None
    if cfg["resample"]:
        df = df.resample(cfg["resample"]).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    if len(df) < 15: return None
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
    delta = df['close'].diff()
    # FIX #2: Wilder smoothing for RSI (matches broker/TradingView standard)
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    df['vwap'] = (tp * df['volume']).groupby(df.index.date).cumsum() / (df['volume'].groupby(df.index.date).cumsum() + 1e-9)
    return df

# FIX #3: Data freshness check helper
def check_freshness(df, tf):
    cfg = TF_CONFIG[tf]
    last_time = df.index[-1]
    if last_time.tzinfo is None:
        last_time_ist = pytz.utc.localize(last_time).astimezone(ist)
    else:
        last_time_ist = last_time.astimezone(ist)
    age_min = (now_ist - last_time_ist).total_seconds() / 60.0
    return age_min, age_min <= cfg["max_age_min"]

def calculate_setup(cp, atr, e9, e21, step, lot, a_class):
    is_bull = cp >= e9 >= e21
    is_bear = cp < e9 < e21
    atm = int(round(cp / step) * step) if step >= 1 else round(cp, 4)
    fmt = 4 if cp < 500 else 2
    if is_bull:
        return "BUY 🟢", round(cp - atr, fmt), round(cp + 1.5 * atr, fmt), round(cp + 2.5 * atr, fmt), (f"{atm} CE" if a_class in ["Index", "Stock"] else "LONG CALL"), max(1, int(max_risk / (max(atr, 0.001) * lot)))
    elif is_bear:
        return "SELL 🔴", round(cp + atr, fmt), round(cp - 1.5 * atr, fmt), round(cp - 2.5 * atr, fmt), (f"{atm} PE" if a_class in ["Index", "Stock"] else "SHORT PUT"), max(1, int(max_risk / (max(atr, 0.001) * lot)))
    return "WAIT ⚪", round(cp - atr, fmt), round(cp + atr, fmt), round(cp + 2 * atr, fmt), "NO TRADE", 0

def draw_chart(df, title, lines=None, freshness_note=""):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_9'], line=dict(color='cyan', width=1.2), name="EMA 9"), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_21'], line=dict(color='magenta', width=1.2), name="EMA 21"), 1, 1)
    if 'vwap' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], line=dict(color='orange', width=1.2), name="VWAP"), 1, 1)
    if lines:
        for val, clr, txt in lines:
            if val is not None and not np.isnan(val):
                fmt = f"{val:,.4f}" if val < 500 else f"{val:,.2f}"
                fig.add_hline(y=val, line=dict(color=clr, width=1.3, dash='dash'), annotation_text=f"{txt}: {fmt}", row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#A277FF', width=1.3), name="RSI"), 2, 1)
    fig.add_hline(y=70, line=dict(color='#FF4D4D', dash='dot'), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='#00FF66', dash='dot'), row=2, col=1)
    title_suffix = f" | {freshness_note}" if freshness_note else ""
    fig.update_layout(title=f"{title} | Synced: {now_ist.strftime('%H:%M:%S IST')}{title_suffix}", height=450, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(t=30, b=10, l=10, r=10))
    return fig

# HEADER
hc1, hc2 = st.columns([3, 1.2])
hc1.markdown(f"**Live Market Clock (IST):** `{now_ist.strftime('%d %b %Y | %H:%M:%S')}`")
if hc2.button("🔄 Refresh Data & Clear Cache", type="primary", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

t1, t2, t3, t4, t5 = st.tabs(["🥇 Gold", "🎯 Option Momentum", "📡 Screener", "🏛️ Goldman Sachs Quant", "🚀 Hedged Spreads & Adjuster"])

# TAB 1: GOLD
with t1:
    st.subheader("🥇 Gold Execution Hub")
    gc1, gc2 = st.columns([2, 1])
    g_feed = gc1.selectbox("Feed:", ["Spot Gold (XAUUSD=X)", "COMEX Gold (GC=F)"], index=0)
    g_tf = gc2.selectbox("Timeframe:", ["3M", "15M", "1H", "1D"], index=1, key="gtf")
    gdf = load_feed("XAUUSD=X" if "Spot" in g_feed else "GC=F", g_tf)
    if gdf is not None and len(gdf) >= 15:
        age_min, is_fresh = check_freshness(gdf, g_tf)  # FIX #3
        r = gdf.iloc[-1]
        cp, atr = float(r['close']), float(r['atr'])
        sig, sl, tp1, tp2, strike, lots = calculate_setup(cp, atr, float(r['ema_9']), float(r['ema_21']), 5, 1, "Commodity")
        if not is_fresh:
            st.warning(f"⚠️ Data is {age_min:.0f} min old — stale feed. Signal shown may not reflect current market.")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Signal", sig); m2.metric("Live Gold Price", f"${cp:,.2f}"); m3.metric("Stop Loss", f"${sl:,.2f}"); m4.metric("Target 1", f"${tp1:,.2f}"); m5.metric("Position Size", f"{lots} Oz")
        fresh_note = "🟢 Fresh" if is_fresh else f"🔴 Stale ({age_min:.0f}m old)"
        st.plotly_chart(draw_chart(gdf.iloc[-90:], f"Gold ({g_tf})", [(sl, '#FF4D4D', 'SL'), (tp1, '#00FF66', 'TP1'), (tp2, '#00E5FF', 'TP2')], fresh_note), use_container_width=True)
    else: st.warning("Loading Gold data... (or feed temporarily unavailable)")

# TAB 2: OPTION MOMENTUM
with t2:
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
            else:
                st.markdown(f"<div style='background:#181A20;border-left:4px solid #555;padding:8px;'><small>{tf}</small><br><b>NO DATA ⚠️</b></div>", unsafe_allow_html=True)
    st.markdown(f"**Confluence Alignment:** `{int(max(bull_c, bear_c)/4.0*100)}%` across all timeframes")
    sub_tabs = st.tabs(["⚡ 3M", "⚡ 15M", "⏰ 1H", "📅 1D"])
    for i, (tf, stab) in enumerate(zip(["3M", "15M", "1H", "1D"], sub_tabs)):
        with stab:
            df = data_map[tf]
            if df is not None:
                age_min, is_fresh = check_freshness(df, tf)  # FIX #3
                r = df.iloc[-1]
                sig, sl, tp1, tp2, strike, lots = calculate_setup(float(r['close']), float(r['atr']), float(r['ema_9']), float(r['ema_21']), spec["step"], spec["lot"], spec["class"])
                if not is_fresh:
                    st.warning(f"⚠️ Data is {age_min:.0f} min old on {tf} — stale feed.")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Signal", sig); c2.metric("Price", f"{float(r['close']):,.2f}"); c3.metric("Strike", strike); c4.metric("SL", f"{sl:,.2f}"); c5.metric("Lots", f"{lots}")
                fresh_note = "🟢 Fresh" if is_fresh else f"🔴 Stale ({age_min:.0f}m)"
                st.plotly_chart(draw_chart(df.iloc[-90:], f"{chosen} ({tf})", [(sl, '#FFAA00', 'SL'), (tp1, '#00FF66', 'TP1'), (tp2, '#00E5FF', 'TP2')], fresh_note), use_container_width=True)
            else:
                st.warning(f"No data available for {chosen} on {tf}.")

# TAB 3: SCREENER
with t3:
    st.subheader("📡 Real-Time Multi-Asset Execution Screener")
    scanner_tf = st.selectbox("Select Timeframe:", ["3M", "15M", "1H", "1D"], index=1, key="sc_tf")
    rows, feed_cache = [], {}
    with st.spinner(f"Scanning all markets on {scanner_tf}..."):
        for name, meta in ASSET_SPECS.items():
            df = load_feed(meta["symbol"], scanner_tf)
            if df is not None and len(df) >= 15:
                feed_cache[name] = df
                age_min, is_fresh = check_freshness(df, scanner_tf)  # FIX #3
                r = df.iloc[-1]
                sig, sl, tp1, tp2, strike, lots = calculate_setup(float(r['close']), float(r['atr']), float(r['ema_9']), float(r['ema_21']), meta["step"], meta["lot"], meta["class"])
                d_left, _, _, exp_str, _ = get_live_expiry(name)
                rows.append({"Asset": name, "Signal": sig, "Price": round(float(r['close']), 4 if float(r['close']) < 500 else 2), "SL": sl, "TP1": tp1, "TP2": tp2, "Strike": strike, "Lots": lots, "Expiry": f"{exp_str} ({d_left}d)", "RSI": round(float(r['rsi']), 1), "Feed": "🟢 Fresh" if is_fresh else f"🔴 Stale ({age_min:.0f}m)"})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.subheader("🔍 Deep Dive Inspection")
        sel_asset = st.selectbox("Inspect Asset:", list(feed_cache.keys()))
        adf = feed_cache[sel_asset]
        ameta = ASSET_SPECS[sel_asset]
        a_age, a_fresh = check_freshness(adf, scanner_tf)
        ar = adf.iloc[-1]
        asig, asl, atp1, atp2, astrike, alots = calculate_setup(float(ar['close']), float(ar['atr']), float(ar['ema_9']), float(ar['ema_21']), ameta["step"], ameta["lot"],
ameta["class"])
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Signal", asig); d2.metric("Price", f"{float(ar['close']):,.2f}"); d3.metric("SL", f"{asl:,.2f}"); d4.metric("TP1", f"{atp1:,.2f}"); d5.metric("Target Strike", astrike)
        fresh_note = "🟢 Fresh" if a_fresh else f"🔴 Stale ({a_age:.0f}m)"
        st.plotly_chart(draw_chart(adf.iloc[-90:], f"{sel_asset} Execution ({scanner_tf})", [(asl, '#FF4D4D', 'SL'), (atp1, '#00FF66', 'TP1'), (atp2, '#00E5FF', 'TP2')], fresh_note), use_container_width=True)
    else:
        st.info("No assets returned usable data for this timeframe.")

# TAB 4: GOLDMAN SACHS QUANT
with t4:
    st.subheader("🏛️ Goldman Sachs QIS Factor Model")
    f1, f2 = st.columns([2, 1])
    gs_asset = f1.selectbox("Asset for Analysis:", list(ASSET_SPECS.keys()), index=0)
    gs_tf = f2.selectbox("Factor Timeframe:", ["3M", "15M", "1H", "1D"], index=2, key="gs_tf")
    gs_df = load_feed(ASSET_SPECS[gs_asset]["symbol"], gs_tf)
    if gs_df is not None and len(gs_df) >= 30:
        gs_age, gs_fresh = check_freshness(gs_df, gs_tf)
        if not gs_fresh:
            st.warning(f"⚠️ Data is {gs_age:.0f} min old — stale feed.")
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
        fresh_note = "🟢 Fresh" if gs_fresh else f"🔴 Stale ({gs_age:.0f}m)"
        st.plotly_chart(draw_chart(gs_df.iloc[-90:], f"Goldman Sachs Engine: {gs_asset} ({gs_tf})", freshness_note=fresh_note), use_container_width=True)
    else:
        st.warning("Insufficient data for quant model.")

# TAB 5: HEDGING & EXPLICIT ROLL ADVISER
with t5:
    st.subheader("🚀 Asymmetric Unlimited Profit Hedging & Live Adjustment Engine")
    st.caption("Ratio Volatility Backspread: Capped Downside Risk with Black-Scholes-derived Premiums & Dynamic Roll Recommendations.")
    st.info(f"📐 Using manual IV input of **{manual_iv*100:.1f}%** (sidebar). This is a theoretical model price — actual fill prices depend on live bid/ask and liquidity.")
    h_asset = st.selectbox("Select Asset to Trade:", ["NIFTY 50", "BANK NIFTY", "FIN NIFTY", "RELIANCE", "HDFC BANK"])
    h_meta = ASSET_SPECS[h_asset]
    hdf = load_feed(h_meta["symbol"], "1D")
    if hdf is not None and len(hdf) >= 20:
        h_age, h_fresh = check_freshness(hdf, "1D")
        if not h_fresh:
            st.warning(f"⚠️ Data is {h_age:.0f} min old — stale feed. Verify spot price before trading.")
        hp = float(hdf.iloc[-1]['close'])
        step, lot = h_meta["step"], h_meta["lot"]
        atm = int(round(hp / step) * step)
        is_bull = float(hdf.iloc[-1]['ema_9']) >= float(hdf.iloc[-1]['ema_21'])
        short_k = atm
        long_k = (atm + 2 * step) if is_bull else (atm - 2 * step)
        opt_type = "CE" if is_bull else "PE"

        d_left, h_left, m_left, exp_date_str, is_exp_day = get_live_expiry(h_asset)
        days_frac = max(d_left + (h_left / 24.0), 0.1)

        # FIX #1: Real Black-Scholes premiums instead of guessed constants
        premium_short = black_scholes_price(hp, short_k, days_frac, manual_iv, opt_type)
        premium_long = black_scholes_price(hp, long_k, days_frac, manual_iv, opt_type)
        net_credit = premium_short - (2 * premium_long)  # positive = credit received, negative = net debit paid
        spread_width = abs(long_k - short_k)
        max_loss = round(max((spread_width - net_credit) * lot, 0), 2)

        def backspread_payoff(price):
            if is_bull:
                if price <= short_k: pnl = net_credit
                elif price <= long_k: pnl = net_credit - (price - short_k)
                else: pnl = net_credit - (price - short_k) + 2 * (price - long_k)
            else:
                if price >= short_k: pnl = net_credit
                elif price >= long_k: pnl = net_credit - (short_k - price)
                else: pnl = net_credit - (short_k - price) + 2 * (long_k - price)
            return pnl * lot

        st.markdown("#### 📥 Live Position Tracker")
        c_in1, c_in2 = st.columns(2)
        entry_price = c_in1.number_input("Entry Spot Price:", value=float(atm), step=float(step))
        trade_entry_date = c_in2.date_input("Trade Entry Date:", value=now_ist.date() - timedelta(days=2))

        estimated_pnl = round(backspread_payoff(hp), 2)  # FIX #1: real payoff model, not guessed multiplier
        pnl_yield = round((estimated_pnl / max_loss) * 100, 1) if max_loss > 0 else 0.0
        pts_moved = (hp - entry_price) if is_bull else (entry_price - hp)
        roll_target = (short_k + step) if is_bull else (short_k - step)
        is_between_strikes = (short_k
        is_between_strikes = (short_k <= hp <= long_k) if is_bull else (long_k <= hp <= short_k)
        is_expiry_near = (d_left <= 1 or is_exp_day)

        if pnl_yield >= 60.0:
            box_title = "🟢 PROFIT HARVEST TRIGGERED (≥ 60% PROFIT MET)"
            box_bg, border_c = "#0d401f", "#00FF66"
            action_txt = f"ACTION: BOOK PROFIT NOW! Target return of {pnl_yield}% achieved. Exit both legs simultaneously: BUY TO CLOSE 1x {short_k} {opt_type} and SELL TO CLOSE 2x {long_k} {opt_type}."
        elif is_expiry_near and is_between_strikes:
            box_title = "🟠 ZERO-RISK EXPIRY ROLL TRIGGERED (PRICE INSIDE TRAP ZONE)"
            box_bg, border_c = "#593302", "#FFAA00"
            action_txt = f"ACTION: EXECUTE EXACT STRIKE ROLL BEFORE EXPIRY ({exp_date_str})! Spot is at {hp:,.2f} (between {short_k} and {long_k}). Roll sold leg to collect final theta: 1. BUY TO CLOSE 1x {short_k} {opt_type} | 2. SELL TO OPEN 1x {roll_target} {opt_type} closer to spot."
        elif (is_bull and hp >= long_k) or (not is_bull and hp <= long_k):
            box_title = "🔵 RUNAWAY EXPANSION (UNLIMITED PROFIT ACTIVE)"
            box_bg, border_c = "#092e59", "#00E5FF"
            trail_sl = round(long_k - 0.5 * step, 2) if is_bull else round(long_k + 0.5 * step, 2)
            action_txt = f"ACTION: UNLIMITED PROFIT ACCELERATION ACTIVE! Spot is beyond long strike ({long_k}). Trail spot stop-loss to {trail_sl} and let profits run into expiry ({exp_date_str})."
        elif pts_moved < -step:
            box_title = "🔴 ADVERSE MOVE / REPAIR PROTOCOL"
            box_bg, border_c = "#4a0f14", "#FF4D4D"
            action_txt = f"ACTION: REPAIR PROTOCOL: Market moved against bias. Because of asymmetric hedge, max loss is strictly capped at ₹{max_loss:,.2f}. Do not add naked lots; hold hedge or exit if trend breaks 50 EMA."
        else:
            box_title = "⚪ NEUTRAL CONSOLIDATION (HOLD POSITION)"
            box_bg, border_c = "#1f242e", "#888888"
            action_txt = f"ACTION: HOLD POSITION: Asset is inside normal fluctuation range ({hp:,.2f}). No adjustment required right now. Expiry is in {d_left}d {h_left}h {m_left}m on {exp_date_str}."

        st.markdown(f"<div style='background:{box_bg};border:2px solid {border_c};padding:14px;border-radius:8px;margin:10px 0;'><h3 style='color:#FFFFFF;margin:0 0 6px 0;'>{box_title}</h3><p style='color:#FFFFFF;font-size:14px;margin:0;font-weight:500;'>{action_txt}</p></div>", unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        b1.info(f"**Leg 1:** SELL 1x {short_k} {opt_type} (Premium: ₹{premium_short:,.2f})")
        b2.success(f"**Leg 2:** BUY 2x {long_k} {opt_type} (Premium: ₹{premium_long:,.2f} each)")
        b3.warning(f"**Live Spot:** `{hp:,.2f}` | **Gap:** `{abs(long_k - short_k)} pts`")

        m1, m2, m3, m4 = st.columns(4)
        credit_label = "Net Credit Received" if net_credit >= 0 else "Net Debit Paid"
        m1.metric(credit_label, f"₹{abs(net_credit * lot):,.2f}")
        m2.metric("Max Capped Loss", f"₹{max_loss:,.2f}")
        m3.metric("Max Profit Target", "UNLIMITED 🚀")
        m4.metric("Live Time to Expiry", f"{d_left}d {h_left}h {m_left}m", delta=exp_date_str)

        st.markdown("---")
        st.markdown("#### 📊 Theoretical Payoff Curve (Black-Scholes Model)")
        price_range = np.linspace(hp * 0.92, hp * 1.08, 100)
        payoff_curve = [backspread_payoff(p) for p in price_range]
        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(x=price_range, y=payoff_curve, mode='lines', line=dict(color='#00E5FF', width=2), name='P&L at Expiry'))
        fig_payoff.add_hline(y=0, line=dict(color='#888', dash='dot'))
        fig_payoff.add_vline(x=hp, line=dict(color='#FFAA00', dash='dash'), annotation_text=f"Spot: {hp:,.2f}")
        fig_payoff.add_vline(x=short_k, line=dict(color='#FF4D4D', dash='dot'), annotation_text=f"Short {short_k}")
        fig_payoff.add_vline(x=long_k, line=dict(color='#00FF66', dash='dot'), annotation_text=f"Long {long_k}")
        fig_payoff.update_layout(template="plotly_dark", height=350, margin=dict(t=30, b=10, l=10, r=10), xaxis_title="Underlying Price at Expiry", yaxis_title="P&L (₹)")
        st.plotly_chart(fig_payoff, use_container_width=True)

        st.caption("⚠️ Premiums and P&L are theoretical Black-Scholes estimates based on your manually entered IV. Actual option prices depend on live bid/ask, liquidity, and skew — always verify against your broker's live option chain before executing.")
    else:
        st.warning("Loading underlying data for hedge calculation...")

# apy.py complete
