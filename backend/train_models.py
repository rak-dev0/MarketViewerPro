"""
Offline training entrypoint for MarketViewerPro models.

Usage examples:
    # Single-ticker baseline
    python train_models.py --ticker SPY --period 3y

    # Multi-ticker pooled model
    python train_models.py --tickers SPY,AAPL,MSFT,NVDA,AVGO --period 3y
"""

import argparse
import logging

from app.services.ml_service import (
    train_models,
    train_models_multi,
    get_model_metrics,
)
from app.logging_config import configure_logging

configure_logging()
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ticker",
        default="SPY",
        help="Single ticker for training (ignored if --tickers is provided)",
    )
    parser.add_argument(
        "--tickers",
        help="Comma-separated list for multi-ticker training (e.g. SPY,AAPL,MSFT)",
    )
    parser.add_argument(
        "--period",
        default="2y",
        help="History window (e.g. 1y, 2y, 3y, 5y)",
    )
    args = parser.parse_args()

    if args.tickers:
        tickers = [
            t.strip().upper()
            for t in args.tickers.split(",")
            if t.strip()
        ]
        log.info(f"Training multi-ticker RF on {tickers} for period={args.period}...")
        train_models_multi(tickers=tickers, period=args.period)
    else:
        log.info(
            f"Training single-ticker RF on {args.ticker} for period={args.period}..."
        )
        train_models(ticker=args.ticker, period=args.period)

    metrics = get_model_metrics()
    log.info(f"Model metrics after training: {metrics}")


if __name__ == "__main__":
    main()
