import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime
import pytz
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Professional Trading Dashboard",
    layout="wide",
    page_icon="📈",
)

IST = pytz.timezone("Asia/Kolkata")


def current_ist():
    return datetime.now(IST)


st.title("Professional Trading Dashboard")
st.caption(
    f"Last page load: {current_ist().strftime('%d-%b-%Y %I:%M:%S %p IST')} "
    "| Research and screening tool — not an execution system."
)

# ============================================================
# DATA NORMALIZATION
# ============================================================
def normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance columns, including newer MultiIndex responses."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        # yfinance may return (Price, Ticker) or (Ticker, Price).
        wanted = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        level0 = set(map(str, out.columns.get_level_values(0)))
        level1 = set(map(str, out.columns.get_level_values(1)))

        if wanted.intersection(level0):
            out.columns = out.columns.get_level_values(0)
        elif wanted.intersection(level1):
            out.columns = out.columns.get_level_values(1)
        else:
            out.columns = out.columns.get_level_values(0)

    out = out.loc[:, ~pd.Index(out.columns).duplicated()]
    required = ["Open", "High", "Low", "Close"]
    if not all(col in out.columns for col in required):
        return pd.DataFrame()

    if "Volume" not in out.columns:
        out["Volume"] = 0.0

    out = out[["Open", "High", "Low", "Close", "Volume"]].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.columns = [str(c).lower() for c in out.columns]
    return out


def compute_indicators(raw_df: pd.DataFrame):
    """125-bar breakout indicator engine used by Gold and NSE scanner."""
    df = normalize_yfinance_columns(raw_df)
    if df.empty or len(df) < 130:
        return None

    df["high_125"] = df["high"].shift(1).rolling(125, min_periods=125).max()
    df["low_125"] = df["low"].shift(1).rolling(125, min_periods=125).min()
    df["vol_sma_125"] = df["volume"].rolling(125, min_periods=125).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(14, min_periods=14).mean()
    df["atr_sma_50"] = df["atr"].rolling(50, min_periods=30).mean()

    df["sma_50"] = df["close"].rolling(50, min_periods=50).mean()
    df["sma_200"] = df["close"].rolling(200, min_periods=100).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    result = df.dropna(subset=["high_125", "vol_sma_125", "atr", "atr_sma_50", "sma_50"])
    return result if not result.empty else None


# ============================================================
# CACHED DATA DOWNLOADS
# ============================================================
@st.cache_data(ttl=120, show_spinner=False)
def download_market_data(symbol, period, interval):
    try:
        data = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return normalize_yfinance_columns(data)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_gold_30m():
    # GC=F is COMEX Gold Futures, not spot XAU/USD.
    df = download_market_data("GC=F", "60d", "30m")
    if df.empty or len(df) < 50:
        return None, None, None

    df = df.copy()
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()

    # Session VWAP resets by calendar date in the timestamp timezone returned
    # by yfinance. This is an approximation of the futures trading session.
    session_date = pd.Index(df.index.date)
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    df["tp_volume"] = typical * df["volume"]
    df["cum_tp_volume"] = df["tp_volume"].groupby(session_date).cumsum()
    df["cum_volume"] = df["volume"].groupby(session_date).cumsum()
    df["vwap"] = df["cum_tp_volume"] / df["cum_volume"].replace(0, np.nan)

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=14).mean()

    daily = df.assign(session_date=session_date).groupby("session_date").agg(
        high=("high", "max"), low=("low", "min")
    )
    if len(daily) >= 2:
        pdh = float(daily["high"].iloc[-2])
        pdl = float(daily["low"].iloc[-2])
    else:
        pdh = float(df["high"].max())
        pdl = float(df["low"].min())

    df = df.dropna(subset=["ema_9", "vwap", "atr"])
    return (df if not df.empty else None), pdh, pdl


# ============================================================
# GOLD STRATEGY 1: 125-BAR BREAKOUT
# ============================================================
def build_gold_breakout_data(timeframe_label):
    if timeframe_label.startswith("Daily"):
        return download_market_data("GC=F", "2y", "1d")

    if timeframe_label.startswith("4-Hour"):
        # Yahoo Finance does not consistently provide 4h bars.
        # Download 1h bars and resample into 4h OHLCV candles.
        hourly = download_market_data("GC=F", "730d", "1h")
        if hourly.empty:
            return pd.DataFrame()
        return hourly.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open", "high", "low", "close"])

    if timeframe_label.startswith("1-Hour"):
        return download_market_data("GC=F", "730d", "1h")

    return download_market_data("GC=F", "60d", "15m")


# ============================================================
# NSE UNIVERSE — symbols from the supplied PDF
# ============================================================
NIFTY_UNIVERSE = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
    "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
    "ABB", "AMBUJACEM", "BANKBARODA", "BOSCHLTD", "CANBK",
    "CHOLAFIN", "DABUR", "DLF", "GAIL", "GODREJCP",
    "HAVELLS", "HAL", "ICICIGI", "IOC", "IRCTC", "IRFC",
    "JINDALSTEL", "JIOFIN", "MOTHERSON", "PIDILITIND", "PNB",
    "PFC", "RECLTD", "SIEMENS", "TORNTPHARM", "TVSMOTOR",
    "VEDL", "ZOMATO", "ZYDUSLIFE", "POLYCAB",
]


def analyze_nse_stock(symbol, interval, period):
    try:
        raw = download_market_data(f"{symbol}.NS", period, interval)
        df = compute_indicators(raw)
        if df is None or len(df) < 2:
            return None

        row = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(row["close"])
        breakout_high = float(row["high_125"])
        atr = float(row["atr"])
        vol_avg = float(row["vol_sma_125"])
        vol_ratio = float(row["volume"] / vol_avg) if vol_avg > 0 else 0.0
        pct_from_high = ((close - breakout_high) / breakout_high) * 100

        cond_a = close >= breakout_high
        cond_b = vol_ratio >= 1.0
        cond_c = row["rsi"] < 70
        cond_d = row["atr"] > row["atr_sma_50"]
        cond_e = row["close"] > row["sma_50"]
        cond_f = row["ema_21"] > row["ema_50"]

        is_buy = cond_a and cond_b
        is_sell = (
            row["ema_21"] < row["ema_50"]
            and prev["ema_21"] >= prev["ema_50"]
        )
        is_near = (-3.0 <= pct_from_high < 0.0) and (vol_ratio >= 0.8)
        score = int(sum([cond_a, cond_b, cond_c, cond_d, cond_e, cond_f]))

        if is_buy:
            signal = "BUY"
        elif is_sell:
            signal = "SELL"
        elif is_near:
            signal = "NEAR BUY"
        else:
            signal = "NEUTRAL"

        return {
            "Signal": signal,
            "Stock": symbol,
            "Price": f"₹{close:,.2f}",
            "Breakout High": f"₹{breakout_high:,.2f}",
            "Distance to High": f"{pct_from_high:+.2f}%",
            "Volume Surge": f"{vol_ratio:.2f}x Avg",
            "RSI": round(float(row["rsi"]), 1),
            "Score": f"{score}/6",
            "Stop Loss": f"₹{close - (2 * atr):,.2f}",
            "Target": f"₹{close + (4 * atr):,.2f}",
            "Is_Breakout": bool(is_buy),
            "Is_High_Volume": bool(vol_ratio >= 1.2),
            "Is_Near_Breakout": bool(is_near),
            "Is_Sell": bool(is_sell),
            "Raw_Volume": vol_ratio,
            "Raw_Distance": pct_from_high,
        }
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def run_clean_scan(interval, period):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(analyze_nse_stock, symbol, interval, period): symbol
            for symbol in NIFTY_UNIVERSE
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception:
                continue
    return results


def show_table(rows, columns):
    if rows:
        frame = pd.DataFrame(rows)
        visible = [col for col in columns if col in frame.columns]
        st.dataframe(frame[visible], use_container_width=True, hide_index=True)
    else:
        st.info("No stocks currently meet these conditions.")


# ============================================================
# MAIN NAVIGATION
# ============================================================
gold_tab, nse_tab = st.tabs(["🥇 Gold (COMEX Futures) Dashboard", "🇮🇳 NSE Stocks Scanner"])

# ============================================================
# TAB 1 — GOLD
# ============================================================
with gold_tab:
    st.subheader("Gold Strategy Selection")
    gold_strategy = st.radio(
        "Choose Gold Strategy",
        [
            "Strategy 2: 30M Session VWAP + 9 EMA + PDH/PDL",
            "Strategy 1: 125-Bar Breakout + Volume (Multi-Timeframe)",
        ],
        horizontal=True,
    )

    if st.button("Refresh Gold Data", type="primary", key="refresh_gold"):
        st.cache_data.clear()
        st.rerun()

    if gold_strategy.startswith("Strategy 2"):
        st.markdown("#### Strategy 2 — 30-Minute Session VWAP + 9 EMA + PDH / PDL")
        st.caption(
            "Momentum framework using session VWAP, 9 EMA and previous-session "
            "high/low. Data source: GC=F COMEX Gold Futures."
        )

        with st.spinner("Downloading 30-minute COMEX Gold data..."):
            gold_df, pdh_value, pdl_value = fetch_gold_30m()

        if gold_df is None or len(gold_df) < 2:
            st.error("Could not retrieve enough 30-minute Gold data. Please refresh later.")
        else:
            latest = gold_df.iloc[-1]
            price = float(latest["close"])
            vwap = float(latest["vwap"])
            ema9 = float(latest["ema_9"])
            atr30 = float(latest["atr"])

            bullish = price > vwap and ema9 > vwap
            bearish = price < vwap and ema9 < vwap

            if price > pdh_value:
                pd_status = "Above previous session high — bullish context"
            elif price < pdl_value:
                pd_status = "Below previous session low — bearish context"
            else:
                pd_status = "Trading inside previous session range"

            if bullish:
                signal = "STRONG BUY"
                stop = min(vwap - 1.0, price - (1.5 * atr30))
                target = price + (3.0 * atr30)
            elif bearish:
                signal = "STRONG SELL / SHORT"
                stop = max(vwap + 1.0, price + (1.5 * atr30))
                target = price - (3.0 * atr30)
            else:
                signal = "NEUTRAL"
                stop = np.nan
                target = np.nan

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latest 30M Price", f"${price:,.2f}")
            m2.metric("Signal", signal)
            m3.metric("Stop Loss", f"${stop:,.2f}" if np.isfinite(stop) else "—")
            m4.metric("Target", f"${target:,.2f}" if np.isfinite(target) else "—")

            a, b, c, d = st.columns(4)
            a.metric("Session VWAP", f"${vwap:,.2f}")
            b.metric("9 EMA", f"${ema9:,.2f}")
            c.metric("Previous Session High", f"${pdh_value:,.2f}")
            d.metric("Previous Session Low", f"${pdl_value:,.2f}")
            st.info(f"Previous-session context: {pd_status}")

            chart_df = gold_df.tail(150)
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.04, row_heights=[0.72, 0.28],
            )
            fig.add_trace(
                go.Candlestick(
                    x=chart_df.index,
                    open=chart_df["open"], high=chart_df["high"],
                    low=chart_df["low"], close=chart_df["close"],
                    name="COMEX Gold 30M",
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=chart_df.index, y=chart_df["vwap"],
                           name="Session VWAP", line=dict(width=2)), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=chart_df.index, y=chart_df["ema_9"],
                           name="9 EMA", line=dict(width=1.5)), row=1, col=1
            )
            fig.add_hline(y=pdh_value, line_dash="dash", annotation_text="Previous High",
                          row=1, col=1)
            fig.add_hline(y=pdl_value, line_dash="dash", annotation_text="Previous Low",
                          row=1, col=1)
            fig.add_trace(
                go.Bar(x=chart_df.index, y=chart_df["volume"], name="Volume"),
                row=2, col=1
            )
            fig.update_layout(
                height=720, template="plotly_dark",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("#### Strategy 1 — 125-Bar Breakout + Volume")
        timeframe = st.radio(
            "Select Gold Timeframe",
            [
                "Daily (1D) — Macro Swing",
                "4-Hour (4H) — Swing",
                "1-Hour (1H) — Intraday / Short-Term",
                "15-Minute (15M) — Short-Term",
            ],
            horizontal=True,
        )

        with st.spinner(f"Downloading Gold {timeframe.split('—')[0].strip()} data..."):
            raw_gold = build_gold_breakout_data(timeframe)
            gold_data = compute_indicators(raw_gold)

        if gold_data is None or len(gold_data) < 2:
            st.error(
                "Insufficient Gold candles for the 125-bar strategy. "
                "Yahoo Finance may limit intraday history."
            )
        else:
            row = gold_data.iloc[-1]
            price = float(row["close"])
            breakout_high = float(row["high_125"])
            atr = float(row["atr"])
            volume_avg = float(row["vol_sma_125"])
            volume_ratio = float(row["volume"] / volume_avg) if volume_avg > 0 else 0.0

            cond_breakout = price >= breakout_high
            cond_volume = volume_ratio >= 1.0
            cond_rsi = row["rsi"] < 70
            cond_atr = row["atr"] > row["atr_sma_50"]
            cond_sma = row["close"] > row["sma_50"]
            cond_ema = row["ema_21"] > row["ema_50"]
            bearish_ema = row["ema_21"] < row["ema_50"]
            score = int(sum([
                cond_breakout, cond_volume, cond_rsi,
                cond_atr, cond_sma, cond_ema
            ]))

            if cond_breakout and cond_volume:
                signal = "STRONG BUY"
                stop = price - (2 * atr)
                target = price + (4 * atr)
            elif bearish_ema:
                signal = "SELL / SHORT"
                stop = price + (2 * atr)
                target = price - (4 * atr)
            else:
                signal = "NEUTRAL"
                stop = np.nan
                target = np.nan

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latest Gold Price", f"${price:,.2f}")
            m2.metric("Signal", signal)
            m3.metric("Stop Loss", f"${stop:,.2f}" if np.isfinite(stop) else "—")
            m4.metric("Target", f"${target:,.2f}" if np.isfinite(target) else "—")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("125-Bar High", f"${breakout_high:,.2f}")
            c2.metric("Volume / Average", f"{volume_ratio:.2f}x")
            c3.metric("RSI (14)", f"{row['rsi']:.1f}")
            c4.metric("Confirmation Score", f"{score}/6")

            st.caption(
                "Signal logic: breakout above prior 125-bar high plus volume at/above "
                "its 125-bar average gives BUY. EMA21 below EMA50 gives SELL. "
                "These are screening rules, not validated probabilities."
            )

            chart_data = gold_data.tail(350)
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.04, row_heights=[0.72, 0.28],
            )
            fig.add_trace(
                go.Candlestick(
                    x=chart_data.index,
                    open=chart_data["open"], high=chart_data["high"],
                    low=chart_data["low"], close=chart_data["close"],
                    name="Gold Price",
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=chart_data.index, y=chart_data["high_125"],
                           name="Prior 125-Bar High", line=dict(dash="dash")),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=chart_data.index, y=chart_data["ema_21"],
                           name="21 EMA"), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=chart_data.index, y=chart_data["ema_50"],
                           name="50 EMA"), row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=chart_data.index, y=chart_data["volume"], name="Volume"),
                row=2, col=1
            )
            fig.update_layout(
                height=700, template="plotly_dark",
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2 — NSE STOCK SCANNER
# ============================================================
with nse_tab:
    st.title("🇮🇳 NSE Institutional Volume & Breakout Tracker")
    selected_tf = st.radio(
        "Select Scanner Timeframe",
        [
            "Daily (1D) — Positional Swing",
            "1-Hour (1H) — Short-Term Swing",
        ],
        horizontal=True,
    )

    is_daily = selected_tf.startswith("Daily")
    interval = "1d" if is_daily else "1h"
    period = "2y" if is_daily else "730d"
    bar_label = "125-Day" if is_daily else "125-Hour"

    st.caption(
        f"Rules: {bar_label} high + volume surge + RSI + ATR + SMA/EMA confirmation "
        f"| Universe: {len(NIFTY_UNIVERSE)} symbols | "
        f"Page load: {current_ist().strftime('%d-%b-%Y %I:%M %p IST')}"
    )

    col_a, col_b = st.columns([4, 1])
    with col_b:
        if st.button("Force Re-Scan NSE", type="primary"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner(f"Scanning {len(NIFTY_UNIVERSE)} NSE symbols..."):
        scan_results = run_clean_scan(interval, period)

    if not scan_results:
        st.warning(
            "No usable scan results were returned. Yahoo Finance may be rate-limiting "
            "requests, or the selected interval may not have enough history."
        )
    else:
        st.success(f"Scan completed. {len(scan_results)} of {len(NIFTY_UNIVERSE)} symbols returned usable data.")

        st.subheader(f"1. Active Breakouts — {bar_label} High + Volume")
        breakouts = [item for item in scan_results if item["Is_Breakout"]]
        show_table(
            breakouts,
            ["Signal", "Stock", "Price", "Breakout High", "Volume Surge",
             "Score", "Stop Loss", "Target", "RSI"],
        )

        st.divider()
        st.subheader("2. Top Volume Movers")
        volume_movers = sorted(
            [item for item in scan_results if item["Is_High_Volume"]],
            key=lambda item: item["Raw_Volume"],
            reverse=True,
        )
        show_table(
            volume_movers,
            ["Signal", "Stock", "Price", "Volume Surge",
             "Distance to High", "RSI", "Score"],
        )

        st.divider()
        st.subheader(f"3. Near-Breakout Radar — Within 3% of {bar_label} High")
        near_breakouts = sorted(
            [item for item in scan_results if item["Is_Near_Breakout"]],
            key=lambda item: item["Raw_Distance"],
            reverse=True,
        )
        show_table(
            near_breakouts,
            ["Signal", "Stock", "Price", "Breakout High",
             "Distance to High", "Volume Surge", "RSI", "Score"],
        )

        st.divider()
        st.subheader("4. Trend Exit Alerts — 21 EMA Crossed Below 50 EMA")
        sell_alerts = [item for item in scan_results if item["Is_Sell"]]
        show_table(sell_alerts, ["Signal", "Stock", "Price", "RSI"])

st.divider()
st.caption(
    "Important: This dashboard uses Yahoo Finance data for research. Quotes may be delayed, "
    "incomplete or unavailable. COMEX GC=F is a futures proxy, not broker-grade XAU/USD spot "
    "pricing. Validate every setup independently. No orders are placed by this application."
)
