"""
Kronos Forecast App - app_kronos.py
====================================
Streamlit app that fetches Groww data and runs Kronos AI forecasting
to predict the next N candles. Confirms directional bias for trading.

Port: 8505

Model loading:
  - Tries local path first (./models/) for offline use
  - Falls back to HuggingFace download (needs internet, no firewall block)
  Set HF_HOME env var or place weights in ./models/ to run offline.
"""
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from growwapi import GrowwAPI

warnings.filterwarnings("ignore")

# Add Kronos model path
sys.path.append(os.path.dirname(__file__))

st.set_page_config(page_title="Kronos Forecast", page_icon="🔮", layout="wide")
st.title("Kronos AI Forecast")
st.caption("Foundation-model candle forecasting using Groww data")

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.header("Configuration")

PRESETS = {
    "BANKNIFTY FUT": "BANKNIFTY26JUNFUT",
    "NIFTY FUT": "NIFTY26JUNFUT",
    "SENSEX FUT": "SENSEX26JUNFUT",
    "GOLD FUT": "GOLD05AUG26FUT",
    "SILVER FUT": "SILVER03JUL26FUT",
    "Custom": "",
}
preset = st.sidebar.selectbox("Instrument", list(PRESETS.keys()))
if preset == "Custom":
    symbol = st.sidebar.text_input("Trading Symbol", "BANKNIFTY26JUNFUT")
else:
    symbol = st.sidebar.text_input("Symbol", PRESETS[preset])

exchange_choice = st.sidebar.selectbox("Exchange", ["NSE", "BSE", "MCX"], index=0)
timeframe = st.sidebar.selectbox("Timeframe", ["5m", "15m", "30m", "1h"], index=1)
pred_len = st.sidebar.slider("Candles to Forecast", 4, 48, 12)
lookback_bars = st.sidebar.slider("Context Bars (history)", 100, 512, 400)

st.sidebar.markdown("---")
st.sidebar.subheader("Model")
model_size = st.sidebar.selectbox("Kronos Model", ["Kronos-small", "Kronos-base", "Kronos-mini"], index=0)
temperature = st.sidebar.slider("Temperature (randomness)", 0.5, 1.5, 1.0, 0.1)
sample_count = st.sidebar.slider("Forecast Paths (averaged)", 1, 5, 1)

MODEL_MAP = {
    "Kronos-mini": ("NeoQuasar/Kronos-Tokenizer-2k", "NeoQuasar/Kronos-mini"),
    "Kronos-small": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-small"),
    "Kronos-base": ("NeoQuasar/Kronos-Tokenizer-base", "NeoQuasar/Kronos-base"),
}

# =============================================================================
# MODEL LOADING (cached)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_kronos(model_size):
    from model import Kronos, KronosTokenizer, KronosPredictor
    tok_id, model_id = MODEL_MAP[model_size]

    # Try local models dir first (offline)
    local_dir = os.path.join(os.path.dirname(__file__), "models")
    local_tok = os.path.join(local_dir, tok_id.split("/")[-1])
    local_model = os.path.join(local_dir, model_id.split("/")[-1])

    if os.path.exists(local_tok) and os.path.exists(local_model):
        tokenizer = KronosTokenizer.from_pretrained(local_tok)
        model = Kronos.from_pretrained(local_model)
        source = "local"
    else:
        tokenizer = KronosTokenizer.from_pretrained(tok_id)
        model = Kronos.from_pretrained(model_id)
        source = "huggingface"

    predictor = KronosPredictor(model, tokenizer, max_context=512)
    return predictor, source


# =============================================================================
# DATA FETCH (with cache)
# =============================================================================
def fetch_groww_data(symbol, exchange_choice, timeframe, groww):
    TF_MAP = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}
    interval_min = TF_MAP[timeframe]

    EXMAP = {"NSE": groww.EXCHANGE_NSE, "BSE": groww.EXCHANGE_BSE, "MCX": groww.EXCHANGE_MCX}
    SEGMAP = {"NSE": groww.SEGMENT_FNO, "BSE": groww.SEGMENT_FNO, "MCX": groww.SEGMENT_COMMODITY}
    exch = EXMAP[exchange_choice]
    seg = SEGMAP[exchange_choice]

    end_dt = datetime.combine(date.today(), datetime.strptime("23:30", "%H:%M").time())
    start_dt = datetime.combine(date.today() - timedelta(days=30), datetime.strptime("09:00", "%H:%M").time())

    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol}_{timeframe}_kronos.csv")

    cached_df = None
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file, parse_dates=["timestamps"])
        except:
            cached_df = None

    fetch_start = (cached_df["timestamps"].max() + timedelta(minutes=1)) if cached_df is not None else start_dt

    rows = []
    if fetch_start < end_dt:
        cur = fetch_start
        while cur < end_dt:
            ce = min(cur + timedelta(days=30), end_dt)
            try:
                r = groww.get_historical_candle_data(
                    trading_symbol=symbol, exchange=exch, segment=seg,
                    start_time=cur.strftime("%Y-%m-%d %H:%M:%S"),
                    end_time=ce.strftime("%Y-%m-%d %H:%M:%S"),
                    interval_in_minutes=interval_min)
                for c in r.get("candles", []):
                    rows.append({"timestamps": datetime.fromtimestamp(c[0]), "open": c[1], "high": c[2], "low": c[3], "close": c[4], "volume": c[5] or 0})
            except:
                pass
            cur = ce
            time.sleep(0.2)

    if rows:
        new_df = pd.DataFrame(rows)
        new_df["timestamps"] = pd.to_datetime(new_df["timestamps"])
        df = pd.concat([cached_df, new_df], ignore_index=True) if cached_df is not None else new_df
    elif cached_df is not None:
        df = cached_df
    else:
        return None

    df = df.drop_duplicates(subset=["timestamps"]).sort_values("timestamps").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return df


# =============================================================================
# MAIN
# =============================================================================
if st.sidebar.button("Run Forecast", type="primary", use_container_width=True):
    load_dotenv("C:/Users/10857442/Documents/growwapi/.env")
    API_KEY = os.getenv("GROWW_API_KEY")
    API_SECRET = os.getenv("GROWW_API_SECRET")
    if not API_KEY or not API_SECRET:
        st.error("Missing GROWW_API_KEY / GROWW_API_SECRET in .env")
        st.stop()

    # Fetch data
    with st.spinner("Fetching Groww data..."):
        token = GrowwAPI.get_access_token(api_key=API_KEY, secret=API_SECRET)
        groww = GrowwAPI(token)
        df = fetch_groww_data(symbol, exchange_choice, timeframe, groww)

    if df is None or len(df) < 50:
        st.error(f"Not enough data for {symbol}. Got {0 if df is None else len(df)} bars.")
        st.stop()

    df["amount"] = df["close"] * df["volume"]
    st.success(f"Fetched {len(df)} candles | Last: {df['timestamps'].iloc[-1]}")

    # Load model
    with st.spinner(f"Loading {model_size} (downloads on first run)..."):
        try:
            predictor, source = load_kronos(model_size)
            st.info(f"Model loaded from: {source}")
        except Exception as e:
            st.error(f"Could not load Kronos model: {e}")
            st.warning("""
            **Model download failed.** This usually means HuggingFace is blocked by your network firewall.

            **To fix:**
            1. Run this app on a network without the firewall (home PC), OR
            2. Manually download the weights and place them in `Kronos/models/`:
               - `Kronos-Tokenizer-base/` and `Kronos-small/`
               - Download from https://huggingface.co/NeoQuasar
            """)
            st.stop()

    # Prepare inputs
    lookback = min(lookback_bars, len(df))
    x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_timestamp = df.iloc[-lookback:]["timestamps"].reset_index(drop=True)

    TF_MAP = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}
    freq = timedelta(minutes=TF_MAP[timeframe])
    last_ts = df["timestamps"].iloc[-1]
    y_timestamp = pd.Series([last_ts + freq * (i + 1) for i in range(pred_len)])

    # Forecast
    with st.spinner(f"Forecasting next {pred_len} candles..."):
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=pred_len, T=temperature, top_p=0.9, sample_count=sample_count, verbose=False)
        pred_df.index = y_timestamp

    current_price = df["close"].iloc[-1]
    final_pred = pred_df["close"].iloc[-1]
    total_chg = final_pred - current_price
    pct_chg = total_chg / current_price * 100

    # --- DISPLAY ---
    st.markdown("---")
    st.header("Forecast Result")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"{current_price:,.1f}")
    col2.metric("Forecast (final)", f"{final_pred:,.1f}", f"{total_chg:+.1f}")
    col3.metric("Expected Move", f"{pct_chg:+.2f}%")
    direction = "BULLISH" if total_chg > 0 else "BEARISH"
    col4.metric("Direction", direction)

    if total_chg > 0:
        st.success(f"**Kronos predicts BULLISH** — price rising to {final_pred:,.0f} ({pct_chg:+.2f}%) over next {pred_len} candles")
    else:
        st.error(f"**Kronos predicts BEARISH** — price falling to {final_pred:,.0f} ({pct_chg:+.2f}%) over next {pred_len} candles")

    # Chart: history + forecast
    st.subheader("Price: History + Forecast")
    hist = df.tail(60)[["timestamps", "close"]].copy()
    hist = hist.rename(columns={"close": "Historical"}).set_index("timestamps")
    fc = pred_df[["close"]].rename(columns={"close": "Forecast"})
    fc.index.name = "timestamps"
    combined = pd.concat([hist, fc], axis=0)
    st.line_chart(combined)

    # Forecast table
    st.subheader("Forecasted Candles")
    show_df = pred_df[["open", "high", "low", "close", "volume"]].copy()
    show_df["change_from_now"] = show_df["close"] - current_price
    st.dataframe(show_df.style.format("{:,.1f}"), use_container_width=True)

    # Forecast stats
    st.subheader("Forecast Range")
    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast High", f"{pred_df['high'].max():,.1f}")
    c2.metric("Forecast Low", f"{pred_df['low'].min():,.1f}")
    c3.metric("Range", f"{pred_df['high'].max() - pred_df['low'].min():,.1f}")

    # Trading interpretation
    st.markdown("---")
    st.header("Trading Interpretation")
    forecast_high = pred_df["high"].max()
    forecast_low = pred_df["low"].min()
    if total_chg > 0:
        st.markdown(f"""
        - **Bias:** LONG — Kronos sees upside to ~{final_pred:,.0f}
        - **Entry zone:** Near current {current_price:,.0f} or on dips to {forecast_low:,.0f}
        - **Target:** {forecast_high:,.0f}
        - **Stop:** Below {forecast_low:,.0f}
        - **Use with your signal apps:** If 8502 also shows BUY signals, confluence is strong
        """)
    else:
        st.markdown(f"""
        - **Bias:** SHORT — Kronos sees downside to ~{final_pred:,.0f}
        - **Entry zone:** Near current {current_price:,.0f} or on rallies to {forecast_high:,.0f}
        - **Target:** {forecast_low:,.0f}
        - **Stop:** Above {forecast_high:,.0f}
        - **Use with your signal apps:** If 8502 also shows SELL signals, confluence is strong
        """)

    st.caption("Note: Kronos is a probabilistic forecast, not a guarantee. Use as confluence with your indicator-based signals, not as a sole entry trigger.")

else:
    st.info("Configure and click **Run Forecast** in the sidebar.")
    st.markdown("""
    ### What this app does
    - Fetches your instrument's candles from Groww (cached)
    - Runs **Kronos** — an open-source foundation model trained on 45+ exchanges' candle data
    - Predicts the next N candles (OHLCV)
    - Gives a directional bias (BULLISH/BEARISH) to confirm your indicator signals

    ### First-run requirements
    1. **Kronos model weights** download from HuggingFace (~100MB for Kronos-small)
       - If your network blocks HuggingFace, run on home PC or place weights in `Kronos/models/`
    2. **PyTorch** (already installed in this venv)

    ### How to use with your other apps
    1. Check **8502** for indicator signals (BUY/SELL)
    2. Run **Kronos forecast** here for AI directional bias
    3. **Trade when both agree** (confluence) — much higher probability

    ### Models
    | Model | Params | Speed |
    |-------|--------|-------|
    | Kronos-mini | 4.1M | Fastest |
    | Kronos-small | 24.7M | Balanced (recommended) |
    | Kronos-base | 102M | Most accurate, slower on CPU |
    """)
