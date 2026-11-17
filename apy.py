import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures

st.set_page_config(page_title="NSE NIFTY 500 Breakout Screener", layout="wide", page_icon="📈")

st.title("📈 Institutional Breakout & Trend Screener (NIFTY 500)")
st.caption("Strategy: 125D High Breakout + Volume Surge + RSI < 70 + High Volatility + SMA 50/200 + EMA 21/50")

# --- Fetch NIFTY 500 Constituents Dynamically ---
@st.cache_data(ttl=86400) # Cache list for 24 hours
def get_nifty_500_symbols():
    url = "https://raw.githubusercontent.com/anirudhshenoy/nifty500-list/main/nifty500.csv"
    try:
        df = pd.read_csv(url)
        symbols = df['Symbol'].str.strip().tolist()
        return sorted(list(set(symbols)))
    except:
        # Robust fallback list if GitHub raw is temporarily unreachable
        return [
            "TATAMOTORS", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL",
            "ADANIENT", "TITAN", "ITC", "KOTAKBANK", "LT", "BAJFINANCE", "MARUTI", "COALINDIA",
            "ZOMATO", "SUZLON", "BEL", "HAL", "TRENT", "IRFC", "JIOFIN", "VEDL", "BHEL"
        ]

NIFTY_500 = get_nifty_500_symbols()

tab1, tab2 = st.tabs(["🔍 Inspect Single Stock", "🚀 Scan NIFTY 500 (Market Scanner)"])

# Common function to calculate all technical rules A through K
def process_stock_data(df):
    if df.empty or len(df) < 200:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.rename(columns=str.lower, inplace=True)

    # Rule A: 125D High Breakout (Prior 125 sessions)
    df['high_125'] = df['high'].shift(1).rolling(125).max()
    # Rule B: 125D Average Volume
    df['vol_sma_125'] = df['volume'].rolling(125).mean()
    # Rule C: RSI 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    # Rule D: Volatility ATR 14 vs 50 ATR SMA
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_sma_50'] = df['atr'].rolling(50).mean()
    # Rules E-I: 50 & 200 SMA
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    # Rules J-K: 21 & 50 EMA
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    return df.dropna()

def analyze_single_record(symbol, df):
    if df is None or len(df) < 2:
        return None
    r = df.iloc[-1]
    pr = df.iloc[-2]

    ca = r['close'] > r['high_125']
    cb = r['volume'] > r['vol_sma_125']
    cc = r['rsi'] < 70
    cd = r['atr'] > r['atr_sma_50']
    chi = (r['close'] > r['sma_50']) and (r['sma_50'] > r['sma_200'])
    cj = r['ema_21'] > r['ema_50']
    sell = (r['ema_21'] < r['ema_50']) and (pr['ema_21'] >= pr['ema_50'])

    score = sum([ca, cb, cc, cd, chi, cj])
    cp = round(r['close'], 2)
    at = round(r['atr'], 2)

    return {
        "Stock": symbol,
        "Price (₹)": cp,
        "Signal": "🟢 STRONG BUY" if score == 6 else ("🔴 SELL" if sell else "HOLD"),
        "Score": f"{score}/6",
        "Stop Loss (₹)": round(cp - (2 * at), 2) if score == 6 else "-",
        "Target (₹)": round(cp + (4 * at), 2) if score == 6 else "-",
        "RSI": round(r['rsi'], 1),
        "125D High": round(r['high_125'], 2),
        "Vol Surge": "PASS" if cb else "FAIL"
    }

# ========================================================
# TAB 1: SINGLE STOCK INSPECTION
# ========================================================
with tab1:
    col_input1, col_input2 = st.columns([1, 2])
    with col_input1:
        selected_stock = st.selectbox("Choose from NIFTY 500:", NIFTY_500, index=0)
    with col_input2:
        custom_ticker = st.text_input("Or enter any custom NSE ticker:").upper().strip()

    stock_symbol = custom_ticker if custom_ticker else selected_stock
    ticker = f"{stock_symbol}.NS" if not stock_symbol.endswith(".NS") else stock_symbol

    @st.cache_data(ttl=300)
    def fetch_single(sym):
        try:
            df = yf.download(sym, period="2y", interval="1d", progress=False)
            return process_stock_data(df)
        except:
            return None

    data = fetch_single(ticker)

    if data is None:
        st.error(f"Could not load data for **{ticker}**. Make sure ticker exists on NSE.")
    else:
        res = analyze_single_record(stock_symbol, data)
        latest = data.iloc[-1]
        
        sig_color = "#00C853" if "BUY" in res["Signal"] else ("#D50000" if "SELL" in res["Signal"] else "#FFAB00")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"₹{res['Price (₹)']:,.2f}")
        m2.markdown(f"**Signal**  \n<h2 style='color:{sig_color};margin:0;'>{res['Signal']}</h2>", unsafe_allow_html=True)
        m3.metric("Calculated Stop-Loss", f"₹{res['Stop Loss (₹)']}" if res['Stop Loss (₹)'] != "-" else "-")
        m4.metric("Calculated Target (1:2 RR)", f"₹{res['Target (₹)']}" if res['Target (₹)'] != "-" else "-")

        st.divider()

        st.subheader(f"Strategy Checklist: {res['Score']} Conditions Met")
        r1, r2, r3 = st.columns(3)
        r4, r5, r6 = st.columns(3)
        r1.success("✅ A: 125D High Breakout" if latest['close'] > latest['high_125'] else f"❌ A: Below 125D High (₹{latest['high_125']:.2f})")
        r2.success("✅ B: High Volume Surge" if latest['volume'] > latest['vol_sma_125'] else "❌ B: Below 125D Avg Volume")
        r3.success(f"✅ C: Safe RSI ({latest['rsi']:.1f} < 70)" if latest['rsi'] < 70 else f"❌ C: Overbought RSI ({latest['rsi']:.1f})")
        r4.success("✅ D: Expanding Volatility" if latest['atr'] > latest['atr_sma_50'] else "❌ D: Normal/Low Volatility")
        r5.success("✅ E-I: Price > 50 SMA > 200 SMA" if (latest['close'] > latest['sma_50'] > latest['sma_200']) else "❌ E-I: Macro Trend Incomplete")
        r6.success("✅ J-K: 21 EMA > 50 EMA" if latest['ema_21'] > latest['ema_50'] else "❌ J-K: 21 EMA < 50 EMA")

        st.divider()

        st.subheader(f"Candlestick & Technical Analysis: {stock_symbol}")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['high_125'], line=dict(color='orange', dash='dash'), name="125D Breakout"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['sma_50'], line=dict(color='yellow', width=1.5), name="50 SMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['sma_200'], line=dict(color='green', width=1.5), name="200 SMA"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['ema_21'], line=dict(color='blue', width=1.5), name="21 EMA (Blue)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['ema_50'], line=dict(color='red', width=1.5), name="50 EMA (Red)"), row=1, col=1)

        vol_colors = ['green' if c >= o else 'red' for c, o in zip(data['close'], data['open'])]
        fig.add_trace(go.Bar(x=data.index, y=data['volume'], marker_color=vol_colors, name="Volume"), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['vol_sma_125'], line=dict(color='purple', width=1.5), name="125D Avg Vol"), row=2, col=1)

        fig.add_trace(go.Scatter(x=data.index, y=data['rsi'], line=dict(color='magenta', width=1.5), name="RSI (14)"), row=3, col=1)
        fig.add_hline(y=70, line=dict(color='red', dash='dot'), row=3, col=1)
        fig.add_hline(y=30, line=dict(color='green', dash='dot'), row=3, col=1)
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# ========================================================
# TAB 2: NIFTY 500 MARKET SCANNER (PARALLEL ENGINE)
# ========================================================
with tab2:
    st.subheader(f"Parallel Scanner: {len(NIFTY_500)} Stocks")
    st.write("This engine scans the full NIFTY 500 index concurrently to find active breakouts and trend setups.")

    scan_universe = st.radio(
        "Choose Scan Depth:",
        ("Top 100 Most Liquid (Faster ~ 20 secs)", "Full NIFTY 500 (~ 60-90 secs)"),
        horizontal=True
    )

    stocks_to_scan = NIFTY_500[:100] if "Top 100" in scan_universe else NIFTY_500

    if st.button("🚀 Start Scanning Index Now", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        def scan_worker(sym):
            try:
                df = yf.download(f"{sym}.NS", period="2y", interval="1d", progress=False)
                clean_df = process_stock_data(df)
                return analyze_single_record(sym, clean_df)
            except:
                return None

        # Multi-threaded download for high speed
        completed = 0
        total = len(stocks_to_scan)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_stock = {executor.submit(scan_worker, s): s for s in stocks_to_scan}
            for future in concurrent.futures.as_completed(future_to_stock):
                res = future.result()
                if res:
                    results.append(res)
                completed += 1
                progress_bar.progress(completed / total)
                status_text.text(f"Analyzed {completed}/{total} stocks...")

        status_text.text("Scan complete!")
        res_df = pd.DataFrame(results)

        if not res_df.empty:
            buys = res_df[res_df['Signal'] == "🟢 STRONG BUY"]
            
            st.divider()
            if not buys.empty:
                st.success(f"🎉 Found {len(buys)} Actionable Breakout Setup(s) in NIFTY 500!")
                st.dataframe(buys, use_container_width=True)
            else:
                st.info("No stocks passed all 6 rules simultaneously today. Breakouts with volume are selective.")

            # Filter option
            show_only_strong = st.checkbox("Show only high-scoring setups (Score 5/6 or 6/6)", value=False)
            if show_only_strong:
                filtered_df = res_df[res_df['Score'].isin(["5/6", "6/6"])]
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.dataframe(res_df, use_container_width=True)
