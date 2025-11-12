import io
import logging
from typing import Literal

import pandas as pd
import requests
import yfinance as yf

from ..config import get_settings
from ..core_exceptions import UpstreamDataError
from ..schemas import Candle, PriceResponse
from .cache_service import cache_key_price, get_cached, set_cached, delete_cached
from .features_service import compute_indicators

log = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Upstream fetchers
# ---------------------------------------------------------------------------

def _fetch_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame | None:
    try:
        data = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if data is None or data.empty:
            # No data for this symbol (typo, delisted, etc.)
            return None
        return data
    except Exception as e:
        emsg = str(e)
        if "no price data found" in emsg.lower():
            # Treat as clean "no data" miss, not a system error
            log.info(f"yfinance: no data for {ticker}: {e}")
            return None
        log.warning(f"yfinance failed for {ticker}: {e}")
        return None



def _fetch_stooq(ticker: str) -> pd.DataFrame | None:
  """
  Simple daily fallback from Stooq.
  Only used if yfinance fails. We intentionally keep this conservative.
  """
  try:
      url = f"https://stooq.pl/q/d/l/?s={ticker.lower()}.us&i=d"
      resp = requests.get(url, timeout=5)
      text = resp.text
      if resp.status_code != 200 or "Date,Open,High,Low,Close,Volume" not in text:
          return None
      df = pd.read_csv(io.StringIO(text))
      if df.empty:
          return None
      df["Date"] = pd.to_datetime(df["Date"])
      df.set_index("Date", inplace=True)
      return df
  except Exception as e:
      log.warning(f"Stooq failed for {ticker}: {e}")
      return None


def _fetch_alpha_vantage(ticker: str) -> pd.DataFrame | None:
  api_key = settings.ALPHAVANTAGE_API_KEY
  if not api_key:
      return None
  try:
      url = (
          "https://www.alphavantage.co/query"
          f"?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}"
          f"&apikey={api_key}&outputsize=compact"
      )
      resp = requests.get(url, timeout=8)
      if resp.status_code != 200:
          return None
      js = resp.json()
      ts = js.get("Time Series (Daily)")
      if not ts:
          return None

      records = []
      for date_str, vals in ts.items():
          records.append(
              {
                  "Date": pd.to_datetime(date_str),
                  "Open": float(vals["1. open"]),
                  "High": float(vals["2. high"]),
                  "Low": float(vals["3. low"]),
                  "Close": float(vals["4. close"]),
                  "Volume": float(vals["6. volume"]),
              }
          )

      if not records:
          return None

      df = pd.DataFrame(records).sort_values("Date").set_index("Date")
      return df
  except Exception as e:
      log.warning(f"Alpha Vantage failed for {ticker}: {e}")
      return None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize upstream data into columns: Open, High, Low, Close, Volume.

    Handles:
    - Standard yfinance single-ticker frames
    - MultiIndex columns (ticker, field) or (field, ticker)
    - Avoids duplicate OHLCV columns by taking the first occurrence
    - Refuses to silently fabricate all-zero data
    """
    if df is None or df.empty:
        raise UpstreamDataError("No data from provider")

    df = df.copy()

    # Handle MultiIndex columns (common with yfinance in some configs)
    if isinstance(df.columns, pd.MultiIndex):
        resolved_cols: list[str] = []
        for col in df.columns:
            chosen = None
            for part in col:
                if part is None:
                    continue
                s = str(part).lower()
                if (
                    s.startswith("open")
                    or s.startswith("high")
                    or s.startswith("low")
                    or s.startswith("close")
                    or s.startswith("adj close")
                    or "volume" in s
                ):
                    chosen = str(part)
                    break
            if chosen is None:
                # Fallback: join tuple so we can safely ignore later if irrelevant
                chosen = "_".join(str(p) for p in col if p is not None)
            resolved_cols.append(chosen)
        df.columns = resolved_cols

    # Normalize labels to strings
    df.columns = [str(c) for c in df.columns]

    # Map various names to canonical OHLCV
    colmap: dict[str, str] = {}
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("open"):
            colmap[c] = "Open"
        elif lc.startswith("high"):
            colmap[c] = "High"
        elif lc.startswith("low"):
            colmap[c] = "Low"
        elif lc.startswith("close") or lc.startswith("adj close"):
            colmap[c] = "Close"
        elif "volume" in lc:
            colmap[c] = "Volume"

    df = df.rename(columns=colmap)

    # Drop exact duplicate columns (keep first) to avoid Close being a DataFrame
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Require at least Close
    if "Close" not in df.columns:
        raise UpstreamDataError("No recognizable OHLCV columns in provider data")

    # Fill missing OHLC from Close; Volume defaults to 0 if missing
    if "Open" not in df.columns:
        df["Open"] = df["Close"]
    if "High" not in df.columns:
        df["High"] = df["Close"]
    if "Low" not in df.columns:
        df["Low"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Reject "all zeros" as invalid data
    if (df[["Open", "High", "Low", "Close"]].to_numpy() != 0).sum() == 0:
        raise UpstreamDataError("Provider returned only zero OHLC data")

    return df


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_price(
    ticker: str,
    range_: str = "1y",
    interval: str = "1d",
) -> PriceResponse:
    ticker = ticker.upper()
    cache_key = cache_key_price(ticker, range_, interval)

    cached = get_cached(cache_key)
    if cached:
        payload, is_stale = cached
        try:
            return _build_price_response_from_payload(
                ticker,
                payload,
                source=payload.get("source", "cache"),
                stale=is_stale,
            )
        except Exception as e:
            log.warning(f"Failed to use cached payload for {ticker}: {e}")
            delete_cached(cache_key)

    # No usable cache → fetch fresh with fallbacks
    return _fetch_and_cache(ticker, range_, interval)



def _fetch_and_cache(
    ticker: str,
    range_: str,
    interval: str,
) -> PriceResponse:
    df = None
    source: str | None = None

    # 1) yfinance
    raw = _fetch_yfinance(ticker, range_, interval)
    if raw is not None:
        try:
            df = _normalize(raw)
            source = "yfinance"
        except UpstreamDataError as e:
            log.warning(f"yfinance unusable for {ticker}: {e}")
            df = None

    # 2) Stooq
    if df is None:
        raw = _fetch_stooq(ticker)
        if raw is not None:
            try:
                df = _normalize(raw)
                source = "stooq"
            except UpstreamDataError as e:
                log.warning(f"Stooq unusable for {ticker}: {e}")
                df = None

    # 3) Alpha Vantage
    if df is None:
        raw = _fetch_alpha_vantage(ticker)
        if raw is not None:
            try:
                df = _normalize(raw)
                source = "alphavantage"
            except UpstreamDataError as e:
                log.warning(f"Alpha Vantage unusable for {ticker}: {e}")
                df = None

    if df is None or source is None:
        raise UpstreamDataError("All data providers failed")

    payload = {
        "records": (
            df.reset_index()
              .rename(columns={"index": "Date"})
              .to_dict(orient="records")
        ),
        "source": source,
    }

    cache_key = cache_key_price(ticker, range_, interval)
    set_cached(cache_key, payload)

    return _build_price_response_from_payload(ticker, payload, source, stale=False)



def _build_price_response_from_payload(
  ticker: str,
  payload: dict,
  source: str,
  stale: bool,
) -> PriceResponse:
  records = payload.get("records", [])
  if not records:
      raise UpstreamDataError("No price records in payload")

  df = pd.DataFrame(records)

  # Support either Date or timestamp field
  if "Date" in df.columns:
      df["timestamp"] = pd.to_datetime(df["Date"])
  elif "timestamp" in df.columns:
      df["timestamp"] = pd.to_datetime(df["timestamp"])
  else:
      raise UpstreamDataError("Missing timestamp field in records")

  df.set_index("timestamp", inplace=True)
  df = _normalize(df)

  indicators = compute_indicators(df)

  candles = [
      Candle(
          timestamp=idx.to_pydatetime(),
          open=float(row["Open"]),
          high=float(row["High"]),
          low=float(row["Low"]),
          close=float(row["Close"]),
          volume=float(row["Volume"]),
      )
      for idx, row in df.iterrows()
  ]

  return PriceResponse(
      ticker=ticker,
      candles=candles,
      indicators=indicators,
      source=source,
      stale=stale,
  )
