from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "MarketViewerPro"
    ENV: str = "dev"

    # External APIs
    ALPHAVANTAGE_API_KEY: str | None = None

    # Cache / Storage
    CACHE_DB_URL: str = "sqlite:///./storage/cache.db"
    CACHE_TTL_SECONDS: int = 300  # 5 min
    CACHE_MAX_AGE_SECONDS: int = 3600  # data older than this marked stale

    MODELS_DIR: str = "models"

    # News
    NEWS_FEEDS: List[str] = [
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^IXIC&region=US&lang=en-US",
        "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        "https://www.theverge.com/rss/index.xml",
    ]

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
