from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


class APIError(BaseModel):
    detail: str
    code: str = "error"


class HealthProviderStatus(BaseModel):
    name: str
    status: Literal["up", "degraded", "down"]
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["up", "degraded", "down"]
    providers: List[HealthProviderStatus]


class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class IndicatorSnapshot(BaseModel):
    sma: dict[str, Optional[float]] = {}
    ema: dict[str, Optional[float]] = {}
    rsi14: Optional[float] = None
    volume_change_1d: Optional[float] = None
    volume_change_5d: Optional[float] = None
    price_to_ma: dict[str, Optional[float]] = {}


class PriceResponse(BaseModel):
    ticker: str
    candles: List[Candle]
    indicators: Optional[IndicatorSnapshot]
    source: str
    stale: bool = False


class PredictionResponse(BaseModel):
    ticker: str
    up_probability: float
    predicted_return_pct: float
    model_name: str
    last_trained_at: Optional[datetime]
    samples_used: Optional[int] = None
    recent_accuracy: Optional[float] = None
    reliability_hint: Optional[str] = None
    note: str = Field(
        default=(
            "Experimental signals only. For research & paper trading. "
            "Not investment advice; no profitability guarantee."
        )
    )


class NewsItem(BaseModel):
    title: str
    source: str
    url: str
    published_at: Optional[datetime]
    summary: Optional[str] = None
    sentiment: float
    relevance_score: float
    tickers: List[str] = []


class NewsResponse(BaseModel):
    items: List[NewsItem]
