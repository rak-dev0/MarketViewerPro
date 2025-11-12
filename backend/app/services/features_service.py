import pandas as pd
import numpy as np
from ..schemas import IndicatorSnapshot


MA_WINDOWS = [5, 10, 20, 50, 200]


def compute_indicators(df: pd.DataFrame) -> IndicatorSnapshot:
    """
    Expects df indexed by datetime with columns:
    ['Open','High','Low','Close','Volume']
    Returns indicators for the **last row**.
    """
    if df.empty:
        return IndicatorSnapshot()

    close = df["Close"]
    vol = df["Volume"]

    sma = {str(w): float(close.rolling(w).mean().iloc[-1]) if len(close) >= w else None
           for w in MA_WINDOWS}
    ema = {str(w): float(close.ewm(span=w, adjust=False).mean().iloc[-1]) if len(close) >= w else None
           for w in MA_WINDOWS}

    # RSI(14)
    window = 14
    if len(close) >= window + 1:
        delta = close.diff()
        up = delta.clip(lower=0).rolling(window).mean()
        down = -delta.clip(upper=0).rolling(window).mean()
        rs = up / (down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        rsi14 = float(rsi.iloc[-1])
    else:
        rsi14 = None

    # Volume changes
    volume_change_1d = None
    volume_change_5d = None
    if len(vol) >= 2 and vol.iloc[-2] > 0:
        volume_change_1d = float((vol.iloc[-1] - vol.iloc[-2]) / vol.iloc[-2] * 100)
    if len(vol) >= 6 and vol.iloc[-6] > 0:
        volume_change_5d = float((vol.iloc[-1] - vol.iloc[-6]) / vol.iloc[-6] * 100)

    # Price to MA
    last_price = float(close.iloc[-1])
    price_to_ma = {}
    for w, v in sma.items():
        if v and v != 0:
            price_to_ma[w] = last_price / v
        else:
            price_to_ma[w] = None

    return IndicatorSnapshot(
        sma=sma,
        ema=ema,
        rsi14=rsi14,
        volume_change_1d=volume_change_1d,
        volume_change_5d=volume_change_5d,
        price_to_ma=price_to_ma,
    )
