"""
Kronos Forecast using Groww data
==================================
Fetches BANKNIFTY/NIFTY futures from Groww, runs Kronos to predict next N candles.

Usage: python kronos_groww_forecast.py [SYMBOL] [TIMEFRAME] [PRED_LEN]
"""
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from growwapi import GrowwAPI

# Kronos model
sys.path.append(os.path.dirname(__file__))
from model import Kronos, KronosTokenizer, KronosPredictor

warnings.filterwarnings("ignore")

# --- CONFIG ---
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "BANKNIFTY26JUNFUT"
TIMEFRAME = sys.argv[2] if len(sys.argv) > 2 else "15m"
PRED_LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 12  # next 12 candles
LOOKBACK_BARS = 400  # historical context for Kronos (max 512)

TF_MAP = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}
interval_min = TF_MAP.get(TIMEFRAME, 15)

# --- Load Groww credentials ---
load_dotenv(dotenv_path="C:/Users/10857442/Documents/growwapi/.env")
API_KEY = os.getenv("GROWW_API_KEY")
API_SECRET = os.getenv("GROWW_API_SECRET")

print(f"Fetching {SYMBOL} ({TIMEFRAME}) from Groww...")
token = GrowwAPI.get_access_token(api_key=API_KEY, secret=API_SECRET)
g = GrowwAPI(token)

# Fetch enough bars for lookback (30 days covers ~ enough 15m bars)
end_dt = datetime.combine(date.today(), datetime.strptime("15:30", "%H:%M").time())
start_dt = datetime.combine(date.today() - timedelta(days=30), datetime.strptime("09:15", "%H:%M").time())

rows = []
cur = start_dt
while cur < end_dt:
    ce = min(cur + timedelta(days=30), end_dt)
    try:
        r = g.get_historical_candle_data(
            trading_symbol=SYMBOL, exchange=g.EXCHANGE_NSE, segment=g.SEGMENT_FNO,
            start_time=cur.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=ce.strftime("%Y-%m-%d %H:%M:%S"),
            interval_in_minutes=interval_min)
        for c in r.get("candles", []):
            rows.append({"timestamps": datetime.fromtimestamp(c[0]), "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5] or 0})
    except Exception as e:
        print(f"  fetch error: {e}")
    cur = ce
    time.sleep(0.2)

df = pd.DataFrame(rows).drop_duplicates(subset=["timestamps"]).sort_values("timestamps").reset_index(drop=True)
df["timestamps"] = pd.to_datetime(df["timestamps"])
df["amount"] = df["close"] * df["volume"]  # Kronos optional column
print(f"Fetched {len(df)} candles")

if len(df) < 50:
    print("Not enough data.")
    sys.exit(1)

# Use last LOOKBACK_BARS as context
lookback = min(LOOKBACK_BARS, len(df))
x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
x_timestamp = df.iloc[-lookback:]["timestamps"].reset_index(drop=True)

# Generate future timestamps
last_ts = df["timestamps"].iloc[-1]
freq = timedelta(minutes=interval_min)
y_timestamp = pd.Series([last_ts + freq * (i + 1) for i in range(PRED_LEN)])

# --- Load Kronos ---
print("Loading Kronos model (downloads on first run)...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

# --- Predict ---
print(f"Forecasting next {PRED_LEN} candles...")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=PRED_LEN,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True,
)

# --- Results ---
current_price = df["close"].iloc[-1]
pred_df.index = y_timestamp

print("\n" + "=" * 55)
print(f"  KRONOS FORECAST: {SYMBOL} ({TIMEFRAME})")
print(f"  Current Price: {current_price:,.1f}")
print("=" * 55)
print(f"\n{'Time':<20} {'Pred Close':>12} {'Change':>10}")
print("-" * 45)
for ts, row in pred_df.iterrows():
    chg = row["close"] - current_price
    print(f"{str(ts)[:16]:<20} {row['close']:>12,.1f} {chg:>+10.1f}")

final_pred = pred_df["close"].iloc[-1]
total_chg = final_pred - current_price
pct_chg = total_chg / current_price * 100

print("\n" + "=" * 55)
if total_chg > 0:
    print(f"  DIRECTION: BULLISH  (+{total_chg:.0f} pts / +{pct_chg:.2f}%)")
else:
    print(f"  DIRECTION: BEARISH  ({total_chg:.0f} pts / {pct_chg:.2f}%)")
print(f"  Forecast high: {pred_df['high'].max():,.1f}")
print(f"  Forecast low:  {pred_df['low'].min():,.1f}")
print("=" * 55)

# Save forecast
pred_df.to_csv(f"kronos_forecast_{SYMBOL}.csv")
print(f"\nSaved to kronos_forecast_{SYMBOL}.csv")
