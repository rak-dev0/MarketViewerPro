# MarketViewerPro

Compact market intelligence dashboard:
- Live OHLCV (multi-source with cache + stale-while-revalidate)
- Baseline ML signals (RF + EMA fallback)
- Curated news + sentiment
- React + FastAPI + Docker

## Stack

Backend:
- FastAPI
- yfinance, Alpha Vantage (optional), SQLite cache
- scikit-learn (RF), joblib
- RSS + VADER sentiment

Frontend:
- React + Vite
- Recharts candlestick adapter
- Dark, trader-friendly UI

## Dev Run

Backend:

```bash

cd backend
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload

Frontend:

cd frontend
npm install
npm run dev

Visit: http://localhost:5173
Docker

docker-compose up --build

Frontend: http://localhost:5173

Backend: http://localhost:8000/docs
Notes

    ML is experimental: for research & paper trading.

    If RF models are not trained, API falls back to a transparent EMA baseline.

    No guarantee of data availability from free upstream providers.