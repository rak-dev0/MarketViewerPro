from typing import Optional, Union

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from .config import get_settings
from .logging_config import configure_logging
from .schemas import (
    APIError,
    HealthProviderStatus,
    HealthResponse,
    NewsResponse,
    PredictionResponse,
    PriceResponse,
)
from .services import cache_service, ml_service, news_service, price_service
from .core_exceptions import UpstreamDataError

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

configure_logging()
settings = get_settings()

app = FastAPI(
    title="MarketViewerPro API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """
    Lightweight system health summary for UI StatusBar + uptime checks.
    """
    providers: list[HealthProviderStatus] = []

    # API
    providers.append(HealthProviderStatus(name="api", status="up"))

    # Cache (SQLite)
    try:
        cache_service.get_sqlite_cache()
        cache_status = "up"
    except Exception:
        cache_status = "down"
    providers.append(HealthProviderStatus(name="cache", status=cache_status))

    # Models: RF with last_trained_at => up, else degraded (EMA baseline only)
    model_metrics = ml_service.get_model_metrics()
    model_name = model_metrics.get("model_name")
    last_trained = model_metrics.get("last_trained_at")
    if model_name and model_name != "ema_baseline" and last_trained:
        model_status = "up"
    else:
        model_status = "degraded"
    providers.append(HealthProviderStatus(name="models", status=model_status))

    # News feeds configured?
    if settings.NEWS_FEEDS and len(settings.NEWS_FEEDS) > 0:
        news_status = "up"
    else:
        news_status = "degraded"
    providers.append(HealthProviderStatus(name="news_feeds", status=news_status))

    # Overall status
    overall = "up"
    if any(p.status == "down" for p in providers):
        overall = "down"
    elif any(p.status == "degraded" for p in providers):
        overall = "degraded"

    return HealthResponse(status=overall, providers=providers)


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

@app.get(
    "/api/price",
    response_model=PriceResponse,
    responses={
        400: {"model": APIError},
        404: {"model": APIError},
        502: {"model": APIError},
    },
)
def get_price_endpoint(
    ticker: str = Query(..., min_length=1, description="Ticker symbol, e.g. AAPL"),
    range_: str = Query(
        "1y",
        alias="range",
        description="yfinance-style range (e.g. 1mo, 6mo, 1y)",
    ),
    interval: str = Query(
        "1d",
        description="yfinance interval (e.g. 1d, 1h). ML pipeline assumes 1d.",
    ),
) -> Union[PriceResponse, JSONResponse]:
    """
    Return OHLCV + indicators for a ticker with cache + multi-provider fallback.

    Behavior:
    - 200: valid data (+ stale/source flags).
    - 404: no data for this ticker (typo/delisted/etc).
    - 502: upstream providers failed in a way that looks like infra, not symbol.
    - 400: unexpected local error (bad params, serialization, etc).
    """
    try:
        return price_service.get_price(
            ticker=ticker,
            range_=range_,
            interval=interval,
        )
    except UpstreamDataError as e:
        msg = (str(e) or "").lower()

        # Heuristic: treat "no data" / "all providers failed" as no-data-for-symbol.
        if (
            "no data" in msg
            or "no price data" in msg
            or "delisted" in msg
            or "all data providers failed" in msg
        ):
            return JSONResponse(
                status_code=404,
                content={
                    "detail": "No price data for this ticker. Check the symbol.",
                    "code": "no_data",
                },
            )

        # Otherwise: upstream infra issue.
        return JSONResponse(
            status_code=502,
            content={
                "detail": "Upstream price provider failed.",
                "code": "upstream_failed",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"Failed to fetch price data: {e}",
                "code": "price_error",
            },
        )


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

@app.get(
    "/api/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": APIError},
        404: {"model": APIError},
        502: {"model": APIError},
    },
)
def get_predict_endpoint(
    ticker: str = Query(..., min_length=1, description="Ticker symbol"),
    range_: str = Query(
        "6mo",
        alias="range",
        description="History window used to build features (e.g. 3mo, 6mo, 1y)",
    ),
    interval: str = Query(
        "1d",
        description="Currently only 1d is supported for the ML pipeline",
    ),
) -> Union[PredictionResponse, JSONResponse]:
    """
    Return conservative next-day signal for a ticker.

    Behavior:
    - Reuses /api/price data path for consistency.
    - If RF model available -> RF-based signal (rf_v1).
    - Else -> explicit EMA baseline fallback.
    - Never throws raw stack traces; always structured + honest.
    """
    # 1) Get price data via same pipeline as /api/price
    try:
        price = price_service.get_price(
            ticker=ticker,
            range_=range_,
            interval=interval,
        )
    except UpstreamDataError as e:
        msg = (str(e) or "").lower()
        if (
            "no data" in msg
            or "no price data" in msg
            or "delisted" in msg
            or "all data providers failed" in msg
        ):
            return JSONResponse(
                status_code=404,
                content={
                    "detail": "No price data for this ticker. Check the symbol.",
                    "code": "no_data",
                },
            )
        return JSONResponse(
            status_code=502,
            content={
                "detail": "Upstream price provider failed.",
                "code": "upstream_failed",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "detail": f"Failed to fetch data for prediction: {e}",
                "code": "predict_price_error",
            },
        )

    if not price.candles:
        return JSONResponse(
            status_code=404,
            content={
                "detail": "No price data for this ticker.",
                "code": "no_data",
            },
        )

    # 2) Build normalized OHLCV DataFrame for ML service
    df = pd.DataFrame(
        [
            {
                "timestamp": c.timestamp,
                "Open": c.open,
                "High": c.high,
                "Low": c.low,
                "Close": c.close,
                "Volume": c.volume,
            }
            for c in price.candles
        ]
    ).set_index("timestamp")

    # 3) Call ML service; on hard failure, fall back to transparent EMA baseline
    try:
        pred = ml_service.predict_for_ticker(ticker=ticker, df=df)
        return pred
    except Exception as e:
        # Last-resort safety: neutral/EMA-style baseline, explicit about failure.
        fallback = ml_service._ema_baseline(  # type: ignore[attr-defined]
            ticker,
            f"ML pipeline error: {e}",
        )
        return fallback


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

@app.get(
    "/api/news",
    response_model=NewsResponse,
)
def get_news_endpoint(
    ticker: Optional[str] = Query(
        None,
        description="Optional: filter/boost by ticker symbol",
    ),
    limit: int = Query(
        50,
        ge=1,
        le=200,
        description="Max number of articles to return",
    ),
    min_sentiment: Optional[float] = Query(
        None,
        ge=-1.0,
        le=1.0,
        description="Optional: filter by minimum sentiment score",
    ),
    source: Optional[str] = Query(
        None,
        description="Optional: case-insensitive source name filter",
    ),
) -> NewsResponse:
    """
    Curated RSS news with VADER sentiment and relevance scoring.
    """
    items = news_service.fetch_news(
        ticker=ticker,
        limit=limit,
        min_sentiment=min_sentiment,
        source_filter=source,
    )
    return NewsResponse(items=items)


# ---------------------------------------------------------------------------
# Metrics (Model Transparency)
# ---------------------------------------------------------------------------

@app.get("/api/metrics")
def get_metrics():
    """
    Expose basic model metadata for transparency & UI display.
    Not a performance guarantee.
    """
    return {
        "model": ml_service.get_model_metrics(),
    }
