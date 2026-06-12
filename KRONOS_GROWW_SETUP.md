# Kronos + Groww Forecasting Setup

This adds AI candle forecasting (Kronos foundation model) on top of your Groww data.

## Files
- `kronos_groww_forecast.py` — CLI: fetch Groww data, forecast next N candles
- `app_kronos.py` — Streamlit app (port 8505) with charts + trading interpretation
- `download_models.py` — One-time model weight downloader (run on unblocked network)

## Setup (on a network WITHOUT corporate firewall, e.g. home PC)

```bash
# 1. Clone (if not already)
git clone https://github.com/dhrushi/Kronos.git
cd Kronos

# 2. Create venv (Python 3.10-3.12)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install growwapi python-dotenv streamlit

# 4. Download model weights (one time, needs HuggingFace access)
python download_models.py

# 5. Create .env with Groww keys (or point to your growwapi/.env)
#    GROWW_API_KEY=...
#    GROWW_API_SECRET=...
```

## Run

**CLI forecast:**
```bash
python kronos_groww_forecast.py BANKNIFTY26JUNFUT 15m 12
```

**Streamlit app:**
```bash
streamlit run app_kronos.py --server.port 8505
```

## Corporate Firewall Note
HuggingFace (huggingface.co) is blocked on the corporate network (403 Forbidden).
The model weights MUST be downloaded on an unblocked network using `download_models.py`.
Once weights are in `./models/`, the app/script run fully offline — no HuggingFace access needed.

## How to use with your trading apps
1. Check **8502** (signals) for indicator BUY/SELL
2. Run **Kronos** here for AI directional bias
3. Trade when BOTH agree (confluence) = higher probability
