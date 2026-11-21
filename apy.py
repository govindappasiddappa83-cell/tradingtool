import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime
import pytz

st.set_page_config(page_title="Trading Dashboard", layout="wide", page_icon="⚡")
IST = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(IST)

def wilder_rsi(close, period=14):
    d = close.diff()
    g = d.where(d > 0, 0).ewm(alpha=1/period, adjust=False).mean()
    l = (-d.where(d < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + (g / (l + 1e-9))))

def wilder_atr(df, period=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def check_fresh(idx, mins):
    t = idx[-1]
    if t.tzinfo is None:
        t = pytz.utc.localize(t)
    age = (datetime.now(pytz.utc) - t.astimezone(pytz.utc)).total_seconds() / 60.0
    return age, age <= max(mins * 2, 30)

def compute_indicators(df):
    if df is None or df.empty or len(df) < 130:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open','High','Low','Close','Volume']].copy()
    df.rename(columns=str.lower, inplace=True)
    df['high_125'] = df['high'].shift(1).rolling(125).max()
    df['vol_sma_125'] = df['volume'].rolling(125).mean()
    df['rsi'] = wilder_rsi(df['close'])
    df['atr'] = wilder_atr(df)
    df['atr_sma_50'] = df['atr'].rolling(50).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    return df.dropna(subset=['high_125','vol_sma_125','rsi','atr'])

def candle_chart(df, title, lines=None, height=600):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72,0.28])
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
    if lines:
        for y, c, n in lines:
            fig.add_trace(go.Scatter(x=df.index, y=df[y], line=dict(color=c, width=1.6), name=n), row=1, col=1)
    vc = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=vc, name="Volume"), row=2, col=1)
    fig.update_layout(height=height, xaxis_rangeslider_visible=False, template="plotly_dark", title=title)
    return fig

main_tab1, main_tab2 = st.tabs(["Gold Dashboard", "NSE Scanner"])

with main_tab1:
    gstrat = st.radio("Choose Gold Strategy:", ("Strategy 2: VWAP+EMA+PDH/PDL", "Strategy 1: 125-Bar Breakout"), horizontal=True)

    if "Strategy 2" in gstrat:
        st.subheader("Gold Strategy 2: 30M Session VWAP + 9 EMA + PDH/PDL")
        if st.button("Refresh 30M Strategy", type="primary"):
            st.cache_data.clear()
            st.rerun()

        @st.cache_data(ttl=120)
        def fetch_30m():
            df = yf.download("GC=F", period="30d", interval="30m", progress=False)
            if df.empty or len(df) < 50:
                return None, None, None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[['Open','High','Low','Close','Volume']].copy()
            df.rename(columns=str.lower, inplace=True)
            df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['date'] = df.index.date
            tp = (df['high'] + df['low'] + df['close']) / 3.0
            df['tpv'] = tp * df['volume']
            df['vwap'] = df.groupby('date')['tpv'].cumsum() / (df.groupby('date')['volume'].cumsum() + 1e-9)
            df['atr'] = wilder_atr(df)
            daily = df.groupby('date').agg({'high':'max','low':'min'})
            if len(daily) >= 2:
                pdh, pdl = float(daily['high'].iloc[-2]), float(daily['low'].iloc[-2])
            else:
                pdh, pdl = float(df['high'].max()), float(df['low'].min())
            return df.dropna(subset=['ema_9','vwap','atr']), pdh, pdl

        df30, pdh, pdl = fetch_30m()
        if df30 is None or len(df30) < 2:
            st.error("Could not fetch 30M Gold data.")
        else:
            age, fresh = check_fresh(df30.index, 30)
            if not fresh:
                st.warning(f"Last candle is {age:.0f} min old - data may be stale.")
            r = df30.iloc[-1]
            price, vwap, ema9, atr = round(r['close'],2), round(r['vwap'],2), round(r['ema_9'],2), round(r['atr'],2)
            bull = price > vwap and ema9 > vwap
            bear = price < vwap and ema9 < vwap

            if price > pdh:
                ctx = "Above Previous Day High (Breakout)"
            elif price < pdl:
                ctx = "Below Previous Day Low (Breakdown)"
            else:
                ctx = "Inside Previous Day Range"

            if bull:
                sig, sl, tgt = "🟢 STRONG BUY", round(price - 1.5*atr,2), round(price + 3*atr,2)
            elif bear:
                sig, sl, tgt = "🔴 STRONG SELL", round(price + 1.5*atr,2), round(price - 3*atr,2)
            else:
                sig, sl, tgt = "⚪ NEUTRAL", None, None

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Gold Price (30M)", f"${price:,.2f}")
            c2.metric("Signal", sig)
            c3.metric("Stop Loss", f"${sl:,.2f}" if sl else "-")
            c4.metric("Target", f"${tgt:,.2f}" if tgt else "-")

            c5,c6,c7,c8 = st.columns(4)
            c5.metric("VWAP", f"${vwap:,.2f}")
            c6.metric("9 EMA", f"${ema9:,.2f}")
            c7.metric("PDH", f"${pdh:,.2f}")
            c8.metric("PDL", f"${pdl:,.2f}")
            st.info("Context: " + ctx)

            plot_df = df30.iloc[-150:]
            fig = candle_chart(plot_df, "Gold 30M Chart", [('vwap','orange','VWAP'),('ema_9','cyan','9 EMA')], 700)
            fig.add_hline(y=pdh, line=dict(color='#00FF66', dash='dash'), row=1, col=1)
            fig.add_hline(y=pdl, line=dict(color='#FF4D4D', dash='dash'), row=1, col=1)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("Gold Strategy 1: 125-Bar Breakout + Volume")
        gtf = st.radio("Timeframe:", ("Daily","4-Hour","1-Hour","15-Minute"), horizontal=True)
        if st.button("Refresh Gold Price", type="primary"):
            st.cache_data.clear()
            st.rerun()

        tfmap = {"Daily":("1d","2y",False,1440), "4-Hour":("1h","120d",True,240), "1-Hour":("1h","60d",False,60), "15-Minute":("15m","30d",False,15)}
        ival, iper, resamp, imin = tfmap[gtf]

        @st.cache_data(ttl=120)
        def fetch_gold(ival, iper, resamp):
            df = yf.download("GC=F", period=iper, interval=ival, progress=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if resamp:
                df = df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            return compute_indicators(df)

        gdf = fetch_gold(ival, iper, resamp)
        if gdf is None or len(gdf) < 2:
            st.error("Could not fetch Gold candles.")
        else:
            age, fresh = check_fresh(gdf.index, imin)
            if not fresh:
                st.warning(f"Last candle is {age:.0f} min old - data may be stale.")
            r = gdf.iloc[-1]
            close, hi125, atr = round(r['close'],2), round(r['high_125'],2), round(r['atr'],2)
            volr = round(r['volume'] / (r['vol_sma_125'] + 1e-9), 2)
            a,b,c,d,e,f = close>=hi125, volr>=1.0, r['rsi']<70, r['atr']>r['atr_sma_50'], r['close']>r['sma_50'], r['ema_21']>r['ema_50']
            bear_ema = r['ema_21'] < r['ema_50']
            score = sum([a,b,c,d,e,f])

            if a and b:
                sig, sl, tgt = "🟢 STRONG BUY", round(close-2*atr,2), round(close+4*atr,2)
            elif bear_ema:
                sig, sl, tgt = "🔴 SELL", round(close+2*atr,2), round(close-4*atr,2)
            else:
                sig, sl, tgt = "⚪ NEUTRAL", None, None

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Gold Price", f"${close:,.2f}")
            c2.metric("Signal", sig)
            c3.metric("Stop Loss", f"${sl:,.2f}" if sl else "-")
            c4.metric("Target", f"${tgt:,.2f}" if tgt else "-")
            st.metric("Confirmation Score", f"{score}/6")

            fig = candle_chart(gdf, "Gold Chart", [('high_125','orange','125 High'),('ema_21','blue','21 EMA'),('ema_50','red','50 EMA')])
            st.plotly_chart(fig, use_container_width=True)

with main_tab2:
    st.title("NSE Volume & Breakout Tracker")
    stf = st.radio("Timeframe:", ("Daily","1-Hour"), horizontal=True)
    is_daily = stf == "Daily"
    interval = "1d" if is_daily else "1h"
    period = "2y" if is_daily else "60d"
    tfmin = 1440 if is_daily else 60

    UNIVERSE = ["ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BEL","BPCL",
    "BHARTIARTL","BRITANNIA","CIPLA","COALINDIA","DRREDDY","EICHERMOT","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE",
    "HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK","ITC","INDUSINDBK","INFY","JSWSTEEL","KOTAKBANK","LT",
    "M&M","MARUTI","NTPC","NESTLEIND","ONGC","POWERGRID","RELIANCE","SBILIFE","SHRIRAMFIN","SBIN",
    "SUNPHARMA","TCS","TATACONSUM","TATAMOTORS","TATASTEEL","TECHM","TITAN","TRENT","ULTRACEMCO","WIPRO",
    "ABB","AMBUJACEM","BANKBARODA","BOSCHLTD","CANBK","CHOLAFIN","DABUR","DLF","GAIL","GODREJCP",
    "HAVELLS","HAL","ICICIGI","IOC","IRCTC","IRFC","JINDALSTEL","JIOFIN","MOTHERSON","PIDILITIND",
    "PNB","PFC","RECLTD","SIEMENS","TORNTPHARM","TVSMOTOR","VEDL","ZOMATO","ZYDUSLIFE","POLYCAB"]

    def analyze(sym, ival, iper):
        try:
            df = yf.download(sym + ".NS", period=iper, interval=ival, progress=False)
            df = compute_indicators(df)
            if df is None or len(df) < 2:
                return None
            r = df.iloc[-1]
            close, hi125, atr = round(r['close'],2), round(r['high_125'],2), round(r['atr'],2)
            volr = round(r['volume'] / (r['vol_sma_125'] + 1e-9), 2)
            pct = round(((close - hi125) / hi125) * 100, 2)
            a,b,c,d,e,f = close>=hi125, volr>=1.0, r['rsi']<70, r['atr']>r['atr_sma_50'], r['close']>r['sma_50'], r['ema_21']>r['ema_50']
            is_buy = a and b
            is_sell = r['ema_21'] < r['ema_50']
            is_near = -3.0 <= pct < 0.0 and volr >= 0.8
            sig = "🟢 BUY" if is_buy else ("🔴 SELL" if is_sell else ("🟡 NEAR" if is_near else "⚪ NEUTRAL"))
            score = sum([c,d,e,f]) + (1 if a else 0) + (1 if b else 0)
            age, fresh = check_fresh(df.index, tfmin)
            return {"Signal":sig, "Stock":sym, "Price":f"₹{close:,.2f}", "High125":f"₹{hi125:,.2f}", "Dist%":f"{pct:+.2f}%",
                    "VolSurge":f"{volr}x", "RSI":round(r['rsi'],1), "Score":f"{score}/6",
                    "SL":f"₹{round(close-2*atr,2):,.2f}", "Target":f"₹{round(close+4*atr,2):,.2f}",
                    "Fresh":"🟢" if fresh else "🔴", "is_buy":is_buy, "is_sell":is_sell, "is_near":is_near,
                    "volr":volr, "pct":pct}
        except Exception:
            return None

    @st.cache_data(ttl=600)
    def run_scan(ival, iper):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(analyze, s, ival, iper): s for s in UNIVERSE}
            res = [f.result() for f in concurrent.futures.as_completed(futs)]
        return [r for r in res if r is not None]

    if st.button("Force Re-Scan", type="primary"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Scanning stocks..."):
        results = run_scan(interval, period)

    st.subheader("1. Active Breakouts")
    bo = [r for r in results if r["is_buy"]]
    if bo:
        st.dataframe(pd.DataFrame(bo)[["Signal","Stock","Price","High125","VolSurge","Score","SL","Target","RSI","Fresh"]], use_container_width=True)
    else:
        st.info("No breakouts currently.")

    st.subheader("2. Top Volume Movers")
    vm = sorted([r for r in results if r["volr"] >= 1.2], key=lambda x: x["volr"], reverse=True)
    if vm:
        st.dataframe(pd.DataFrame(vm)[["Signal","Stock","Price","VolSurge","Dist%","RSI","Score","Fresh"]], use_container_width=True)

    st.subheader("3. Near-Breakout Radar")
    nb = sorted([r for r in results if r["is_near"]], key=lambda x: x["pct"], reverse=True)
    if nb:
        st.dataframe(pd.DataFrame(nb)[["Signal","Stock","Price","High125","Dist%","VolSurge","RSI","Score","Fresh"]], use_container_width=True)

    st.subheader("4. Trend Exit Alerts")
    se = [r for r in results if r["is_sell"]]
    if se:
        st.dataframe(pd.DataFrame(se)[["Signal","Stock","Price","RSI","Fresh"]], use_container_width=True)

# app.py complete
