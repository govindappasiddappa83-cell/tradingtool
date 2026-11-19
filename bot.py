import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# ==============================================================================
# 1. INSTITUTIONAL STRATEGY CONFIGURATION
# ==============================================================================
PORTFOLIO_UNIVERSE = {
    "NIFTY 50": {"symbol": "^NSEI", "lot_size": 25, "type": "index"},
    "BANK NIFTY": {"symbol": "^NSEBANK", "lot_size": 15, "type": "index"},
    "COMEX GOLD": {"symbol": "GC=F", "lot_size": 1, "type": "commodity"},
    "S&P 500": {"symbol": "^GSPC", "lot_size": 1, "type": "index"}
}

# Risk Management Parameters (Goldman Sachs QIS Core)
TARGET_ANNUAL_VOL = 0.15     # 15% Target Volatility
ACCOUNT_EQUITY = 200000.0    # Trading Capital (INR or USD)
MAX_CAPITAL_AT_RISK_PCT = 0.01  # Max 1% Risk per trade
MOMENTUM_LOOKBACK = 20       # Short-term Lookback
TREND_FILTER_DAYS = 50       # Structural Trend Baseline

# Cloud Secrets
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_alert(message: str):
    """Pushes encrypted execution log to personal phone."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] Telegram credentials missing. Printing locally:\n", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[!] Notification dispatch failed: {e}")

# ==============================================================================
# 2. QUANT MATHEMATICAL ENGINES
# ==============================================================================
def clean_market_feed(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    req = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in req):
        return None
    return df[req].dropna().copy()

def compute_gs_quant_signals(df, asset_name, meta):
    if len(df) < (TREND_FILTER_DAYS + 10):
        return None

    # 1. Log Returns & Realized Volatility
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['realized_vol_annual'] = df['log_ret'].rolling(window=20).std() * np.sqrt(252)

    # 2. Normalized Time-Series Momentum (Z-Score)
    df['sma_fast'] = df['close'].rolling(window=MOMENTUM_LOOKBACK).mean()
    df['sma_slow'] = df['close'].rolling(window=TREND_FILTER_DAYS).mean()
    spread = df['sma_fast'] - df['sma_slow']
    spread_std = spread.rolling(window=TREND_FILTER_DAYS).std() + 1e-9
    df['momentum_zscore'] = spread / spread_std

    # 3. Dynamic ATR (True Range Risk Buffer)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()

    cur = df.iloc[-1]
    price = float(cur['close'])
    current_vol = float(cur['realized_vol_annual'])
    z_score = float(cur['momentum_zscore'])
    atr = float(cur['atr'])

    # 4. Volatility Scaling Engine
    vol_scale = TARGET_ANNUAL_VOL / max(current_vol, 0.05)
    vol_scale = np.clip(vol_scale, 0.25, 2.0)

    stop_distance = atr * 1.5
    dollar_risk_cap = ACCOUNT_EQUITY * MAX_CAPITAL_AT_RISK_PCT
    lot_size = meta.get("lot_size", 1)

    calculated_qty = (dollar_risk_cap / (stop_distance + 1e-6)) * vol_scale
    final_lots = max(1, int(calculated_qty / lot_size))

    signal = "NEUTRAL"
    sl_price, tp1_price, tp2_price = 0.0, 0.0, 0.0

    if z_score > 1.25 and cur['close'] > cur['sma_slow']:
        signal = "STRONG BUY"
        sl_price = round(price - stop_distance, 2)
        tp1_price = round(price + (stop_distance * 1.5), 2)
        tp2_price = round(price + (stop_distance * 2.5), 2)
    elif z_score < -1.25 and cur['close'] < cur['sma_slow']:
        signal = "STRONG SHORT/SELL"
        sl_price = round(price + stop_distance, 2)
        tp1_price = round(price - (stop_distance * 1.5), 2)
        tp2_price = round(price - (stop_distance * 2.5), 2)

    return {
        "asset": asset_name,
        "symbol": meta["symbol"],
        "price": price,
        "signal": signal,
        "z_score": round(z_score, 2),
        "realized_vol": round(current_vol * 100, 1),
        "vol_scale": round(vol_scale, 2),
        "sl": sl_price,
        "tp1": tp1_price,
        "tp2": tp2_price,
        "lots": final_lots,
        "lot_size": lot_size
    }

# ==============================================================================
# 3. DISPATCHER & CLOUD WORKER
# ==============================================================================
def run_quant_engine():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')
    print(f"[*] Starting Goldman Sachs Quant Screener at {now}...")

    actionable_signals = []

    for name, meta in PORTFOLIO_UNIVERSE.items():
        try:
            raw_data = yf.download(meta['symbol'], period="60d", interval="1h", progress=False)
            df = clean_market_feed(raw_data)
            if df is None:
                continue

            res = compute_gs_quant_signals(df, name, meta)
            if res and res["signal"] != "NEUTRAL":
                actionable_signals.append(res)
        except Exception as e:
            print(f"[!] Error processing {name}: {e}")

    if actionable_signals:
        msg = f"⚡ *GOLDMAN SACHS QIS QUANT ALERT*\n_Timestamp: {now}_\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        for s in actionable_signals:
            emoji = "🟢" if "BUY" in s['signal'] else "🔴"
            msg += (
                f"{emoji} *{s['asset']}* ({s['symbol']})\n"
                f"• *Signal:* `{s['signal']}`\n"
                f"• *Entry Price:* `{s['price']:,.2f}`\n"
                f"• *Stop-Loss (Risk):* `{s['sl']:,.2f}`\n"
                f"• *Target 1 (1:1.5):* `{s['tp1']:,.2f}`\n"
                f"• *Target 2 (1:2.5):* `{s['tp2']:,.2f}`\n"
                f"• *Realized Vol:* `{s['realized_vol']}%` (Scale: {s['vol_scale']}x)\n"
                f"• *Momentum Z-Score:* `{s['z_score']}`\n"
                f"• *Quant Allocation:* `{s['lots']} Lots` ({s['lots'] * s['lot_size']} units)\n"
                "────────────────────\n"
            )
        send_telegram_alert(msg)
    else:
        print("[*] No statistical deviations exceeded threshold. Market in equilibrium.")

if __name__ == "__main__":
    run_quant_engine()
