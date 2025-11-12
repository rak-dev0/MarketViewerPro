import logging
from datetime import datetime, timezone
from typing import List, Optional

import feedparser
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download as nltk_download

from ..config import get_settings
from ..schemas import NewsItem

log = logging.getLogger(__name__)
settings = get_settings()

# Safe init of VADER
try:
    _sia = SentimentIntensityAnalyzer()
except Exception:
    try:
        nltk_download("vader_lexicon")
        _sia = SentimentIntensityAnalyzer()
    except Exception as e:
        log.warning(f"VADER init failed: {e}")
        _sia = None


def _sentiment(text: str) -> float:
    if not _sia or not text:
        return 0.0
    try:
        return float(_sia.polarity_scores(text)["compound"])
    except Exception:
        return 0.0


def fetch_news(
    ticker: Optional[str] = None,
    limit: int = 50,
    min_sentiment: Optional[float] = None,
    source_filter: Optional[str] = None,
) -> List[NewsItem]:
    feeds = settings.NEWS_FEEDS
    items: dict[tuple[str, str], NewsItem] = {}

    for url in feeds:
        try:
            parsed = feedparser.parse(url)
        except Exception as e:
            log.warning(f"Failed to parse RSS {url}: {e}")
            continue

        for e in parsed.entries:
            title = getattr(e, "title", "").strip()
            link = getattr(e, "link", "").strip()
            if not title or not link:
                continue

            key = (title, link)
            if key in items:
                continue

            summary = (getattr(e, "summary", "") or "").strip()
            published = _parse_published(e)
            text = f"{title}. {summary}"
            sentiment = _sentiment(text)
            src = _extract_source(parsed.feed) or "Unknown"

            rel = 0.1
            tickers = []
            if ticker:
                t_upper = ticker.upper()
                if t_upper in title.upper() or t_upper in summary.upper():
                    rel += 0.6
                    tickers.append(t_upper)

            if any(k in title.lower() for k in ("ai", "chip", "cloud", "semiconductor")):
                rel += 0.2

            item = NewsItem(
                title=title,
                source=src,
                url=link,
                published_at=published,
                summary=summary[:500] or None,
                sentiment=sentiment,
                relevance_score=rel,
                tickers=tickers,
            )
            items[key] = item

    news_list = list(items.values())

    if source_filter:
        sf = source_filter.lower()
        news_list = [n for n in news_list if sf in n.source.lower()]

    if min_sentiment is not None:
        news_list = [n for n in news_list if n.sentiment >= min_sentiment]

    if ticker:
        news_list.sort(
            key=lambda n: (n.relevance_score, n.published_at or datetime.min),
            reverse=True,
        )
    else:
        news_list.sort(key=lambda n: n.published_at or datetime.min, reverse=True)

    return news_list[:limit]


def _parse_published(e) -> Optional[datetime]:
    try:
        if getattr(e, "published_parsed", None):
            return datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        if getattr(e, "updated_parsed", None):
            return datetime(*e.updated_parsed[:6], tzinfo=timezone.utc)
    except Exception:
        return None
    return None


def _extract_source(feed) -> str:
    return getattr(feed, "title", "Unknown")
