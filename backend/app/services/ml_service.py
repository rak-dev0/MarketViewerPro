import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib
import yfinance as yf

from ..config import get_settings
from ..schemas import PredictionResponse
from .features_service import compute_indicators
from .price_service import _normalize as normalize_ohlcv
from ..core_exceptions import UpstreamDataError


log = logging.getLogger(__name__)
settings = get_settings()

MODEL_STATE = {
    "clf_path": None,
    "reg_path": None,
    "last_trained_at": None,
    "samples_used": None,
    "recent_accuracy": None,
    "model_name": "ema_baseline",
}

ML_WINDOWS = (5, 10, 20)


def build_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Given normalized OHLCV (Open, High, Low, Close, Volume) indexed by time,
    build the feature matrix used by both training and inference.

    Returns:
        (features_df, feature_cols)
    """
    feat = df.copy()

    # Rolling means / EMAs
    for w in ML_WINDOWS:
        feat[f"SMA_{w}"] = feat["Close"].rolling(w).mean()
        feat[f"EMA_{w}"] = feat["Close"].ewm(span=w, adjust=False).mean()

    # RSI(14)
    delta = feat["Close"].diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    feat["RSI14"] = 100 - (100 / (1 + rs))

    # Volume (already present, but keep explicitly as a feature)
    # Ensure no NaNs
    feat = feat.dropna()

    feature_cols = [
        c
        for c in feat.columns
        if c.startswith("SMA_")
        or c.startswith("EMA_")
        or c.startswith("RSI")
        or c == "Volume"
    ]

    return feat, feature_cols

def get_expected_feature_cols() -> list[str]:
    """
    Canonical feature columns used by RF models.
    Must match what build_ml_features produces.
    """
    cols = []
    for w in ML_WINDOWS:
        cols.append(f"SMA_{w}")
        cols.append(f"EMA_{w}")
    cols.append("RSI14")
    cols.append("Volume")
    return cols


def _models_dir() -> Path:
    p = Path(settings.MODELS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def train_models(
    ticker: str = "SPY",
    period: str = "2y",
) -> None:
    """
    Minimal offline training script.

    Uses the same OHLCV normalization logic as the live price service,
    to avoid shape mismatches from yfinance (MultiIndex, etc).

    Conservative: research / paper trading only.
    """
    log.info(f"Downloading data for {ticker} period={period}...")
    raw = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if raw is None or raw.empty:
        log.warning("No data for model training.")
        return

    try:
        df_ohlcv = normalize_ohlcv(raw)
    except UpstreamDataError as e:
        log.warning(f"Failed to normalize training data: {e}")
        return

    # Build features
    feat, feature_cols = build_ml_features(df_ohlcv)
    if feat.empty or not feature_cols:
        log.warning("No features after preprocessing; aborting training.")
        return

    # Targets
    feat["Return"] = feat["Close"].pct_change().shift(-1)
    feat["Up"] = (feat["Return"] > 0).astype(int)
    feat = feat.dropna(subset=["Return", "Up"])

    if len(feat) < 300:
        log.warning(f"Not enough rows to train RF robustly (got {len(feat)}).")
        return

    X = feat[feature_cols].values
    y_clf = feat["Up"].values
    y_reg = feat["Return"].values


    # Time-series CV for a sanity accuracy estimate
    tscv = TimeSeriesSplit(n_splits=5)
    accs = []
    for tr, te in tscv.split(X):
        clf_cv = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            n_jobs=-1,
            random_state=42,
        )
        clf_cv.fit(X[tr], y_clf[tr])
        preds = clf_cv.predict(X[te])
        accs.append(accuracy_score(y_clf[te], preds))

    # Final models on full data
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X, y_clf)

    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        n_jobs=-1,
        random_state=42,
    )
    reg.fit(X, y_reg)

    models_dir = _models_dir()
    clf_path = models_dir / "rf_clf_v1.joblib"
    reg_path = models_dir / "rf_reg_v1.joblib"

    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)

    MODEL_STATE.update(
        {
            "clf_path": str(clf_path),
            "reg_path": str(reg_path),
            "last_trained_at": datetime.utcnow().isoformat() + "Z",
            "samples_used": int(len(feat)),
            "recent_accuracy": float(np.mean(accs)) if accs else None,
            "model_name": "rf_v1",
            "feature_cols": feature_cols,
        }
    )


    log.info(f"Trained RF models: {MODEL_STATE}")

def train_models_multi(
    tickers: list[str],
    period: str = "3y",
) -> None:
    """
    Train RF models on pooled OHLCV features from multiple tickers.

    - Uses same feature pipeline as single-ticker training.
    - Concatenates per-ticker samples; one global RF model (rf_v1).
    - If no usable data -> exits cleanly without touching existing models.
    """
    frames: list[pd.DataFrame] = []

    for t in tickers:
        ticker = t.upper()
        log.info(f"[multi] Downloading {ticker} period={period}...")
        raw = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if raw is None or raw.empty:
            log.warning(f"[multi] No data for {ticker}, skipping.")
            continue

        try:
            df_ohlcv = normalize_ohlcv(raw)
        except UpstreamDataError as e:
            log.warning(f"[multi] Failed to normalize {ticker}: {e}, skipping.")
            continue

        feat, feature_cols_local = build_ml_features(df_ohlcv)
        if feat.empty or not feature_cols_local:
            log.warning(f"[multi] No features for {ticker}, skipping.")
            continue

        # Targets
        feat["Return"] = feat["Close"].pct_change().shift(-1)
        feat["Up"] = (feat["Return"] > 0).astype(int)
        feat["ticker"] = ticker
        feat = feat.dropna(subset=["Return", "Up"])

        if len(feat) < 100:
            log.warning(f"[multi] Too few rows for {ticker} (got {len(feat)}), skipping.")
            continue

        frames.append(feat)

    if not frames:
        log.warning("[multi] No usable data from any ticker. Leaving existing model unchanged.")
        return

    # Concatenate all tickers' samples
    all_df = pd.concat(frames, axis=0)

    feature_cols = [
        c
        for c in all_df.columns
        if c.startswith("SMA_")
        or c.startswith("EMA_")
        or c.startswith("RSI")
        or c == "Volume"
    ]

    if not feature_cols:
        log.warning("[multi] No feature columns found after concat. Aborting.")
        return

    # Optional: warn if small, but still proceed
    if len(all_df) < 500:
        log.warning(f"[multi] Only {len(all_df)} total samples; model may be weak.")

    # Sort by time for TS-style CV
    all_df = all_df.sort_index()
    X = all_df[feature_cols].values
    y_clf = all_df["Up"].values
    y_reg = all_df["Return"].values

    # TimeSeriesSplit with safe number of splits
    n_splits = 5
    if len(all_df) < (n_splits + 1) * 50:
        n_splits = max(2, min(3, len(all_df) // 100))

    accs = []
    if n_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for tr, te in tscv.split(X):
            clf_cv = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                n_jobs=-1,
                random_state=42,
            )
            clf_cv.fit(X[tr], y_clf[tr])
            preds = clf_cv.predict(X[te])
            accs.append(accuracy_score(y_clf[te], preds))

    # Final RFs on full data
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X, y_clf)

    reg = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        n_jobs=-1,
        random_state=42,
    )
    reg.fit(X, y_reg)

    models_dir = _models_dir()
    clf_path = models_dir / "rf_clf_v1.joblib"
    reg_path = models_dir / "rf_reg_v1.joblib"

    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)

    MODEL_STATE.update(
        {
            "clf_path": str(clf_path),
            "reg_path": str(reg_path),
            "last_trained_at": datetime.utcnow().isoformat() + "Z",
            "samples_used": int(len(all_df)),
            "recent_accuracy": float(np.mean(accs)) if accs else None,
            "model_name": "rf_v1",
            "feature_cols": feature_cols,
        }
    )

    log.info(
        f"[multi] Trained RF on {len(tickers)} tickers, rows={len(all_df)}, "
        f"recent_accuracy={MODEL_STATE['recent_accuracy']}"
    )




def _load_models():
    """
    Lazy-load RF models.

    - First use cached paths in MODEL_STATE if present.
    - Otherwise, look on disk in MODELS_DIR for rf_clf_v1.joblib / rf_reg_v1.joblib.
    - On success, update MODEL_STATE so future calls are cheap.
    """
    clf_path = MODEL_STATE.get("clf_path")
    reg_path = MODEL_STATE.get("reg_path")

    # If not set (new API process), try default locations
    if not clf_path or not reg_path:
        models_dir = _models_dir()
        default_clf = models_dir / "rf_clf_v1.joblib"
        default_reg = models_dir / "rf_reg_v1.joblib"

        if default_clf.exists() and default_reg.exists():
            clf_path = str(default_clf)
            reg_path = str(default_reg)
            MODEL_STATE["clf_path"] = clf_path
            MODEL_STATE["reg_path"] = reg_path
        else:
            return None, None

    try:
        clf = joblib.load(clf_path)
        reg = joblib.load(reg_path)
    except Exception as e:
        log.warning(f"Failed to load RF models from disk: {e}")
        return None, None

    # Stamp minimal metadata if missing
    MODEL_STATE.setdefault("model_name", "rf_v1")
    MODEL_STATE.setdefault("last_trained_at", None)
    # feature_cols will be resolved in predict based on canonical definition

    return clf, reg


    try:
        clf = joblib.load(MODEL_STATE["clf_path"])
        reg = joblib.load(MODEL_STATE["reg_path"])
        return clf, reg
    except Exception as e:
        log.warning(f"Model load failed: {e}")
        return None, None


def get_model_metrics() -> dict:
    """
    Exposed via /api/metrics for transparency.
    """
    return {
        "model_name": MODEL_STATE.get("model_name"),
        "last_trained_at": MODEL_STATE.get("last_trained_at"),
        "samples_used": MODEL_STATE.get("samples_used"),
        "recent_accuracy": MODEL_STATE.get("recent_accuracy"),
    }


def predict_for_ticker(ticker: str, df: pd.DataFrame) -> PredictionResponse:
    """
    Predict next-day direction/return for a given ticker.

    df: normalized OHLCV (Open, High, Low, Close, Volume) indexed by timestamp.
    """
    clf, reg = _load_models()
    if clf is None or reg is None:
        return _ema_baseline(
            ticker,
            "RF model unavailable on disk; using EMA baseline."
        )

    # Build features from recent history
    feat, feat_cols_actual = build_ml_features(df)
    if feat.empty:
        return _ema_baseline(
            ticker,
            "Insufficient data for RF features; using EMA baseline."
        )

    expected_cols = get_expected_feature_cols()

    # Ensure all expected RF features are present
    missing = [c for c in expected_cols if c not in feat.columns]
    if missing:
        # Safer to fallback than to guess feature mapping
        return _ema_baseline(
            ticker,
            f"Feature mismatch ({missing}); using EMA baseline."
        )

    # Align columns in the exact order used for training
    X_last = feat[expected_cols].iloc[[-1]].values

    try:
        proba = float(clf.predict_proba(X_last)[0, 1])
        raw_ret = float(reg.predict(X_last)[0])
    except Exception as e:
        log.warning(f"RF predict error for {ticker}: {e}")
        return _ema_baseline(
            ticker,
            "RF prediction error; using EMA baseline."
        )

    # Clamp predicted next-day return to a sane range (e.g. +/-5%)
    raw_ret = max(min(raw_ret, 0.05), -0.05)

    # Confidence band for stance ONLY (not for zeroing the value)
    upper = 0.55
    lower = 0.45

    # Direction from proba, magnitude from reg
    if proba >= 0.5:
        pred_return = abs(raw_ret)
    else:
        pred_return = -abs(raw_ret)

    # Stance driven by confidence
    if proba > upper:
        stance = "Bullish bias"
    elif proba < lower:
        stance = "Bearish bias"
    else:
        stance = "Neutral / weak edge"

    return PredictionResponse(
        ticker=ticker,
        up_probability=proba,
        predicted_return_pct=pred_return * 100.0,
        model_name=MODEL_STATE.get("model_name", "rf_v1"),
        last_trained_at=MODEL_STATE.get("last_trained_at"),
        reliability_hint=(
            f"Random Forest on OHLCV-derived features; stance from "
            f"P(up) with neutral band [{lower:.2f}, {upper:.2f}]. "
            "Magnitude clamped, research/paper trading only."
        ),
        note=(
            f"{stance}. No guarantee of profitability; validate on out-of-sample data."
        ),
    )




def _ema_baseline(ticker: str, reason: str) -> PredictionResponse:
    """
    Transparent heuristic fallback: EMA slope as weak signal.
    """
    return PredictionResponse(
        ticker=ticker.upper(),
        up_probability=0.5,
        predicted_return_pct=0.0,
        model_name="ema_baseline",
        last_trained_at=None,
        samples_used=None,
        recent_accuracy=None,
        reliability_hint=(
            f"Fallback EMA baseline: neutral/weak signal. {reason}"
        ),
    )
