import React from "react";

export default function Metrics({ prediction }) {
  if (!prediction) {
    return (
      <div className="panel-body">
        <p>Signal will appear here after loading a ticker.</p>
      </div>
    );
  }

  const {
    up_probability,
    predicted_return_pct,
    model_name,
    last_trained_at,
    reliability_hint,
    note,
  } = prediction;

  const upPct = up_probability != null ? (up_probability * 100).toFixed(1) : "–";
  const retPct =
    predicted_return_pct != null ? predicted_return_pct.toFixed(2) : "–";

  let stance = "Neutral";
  if (predicted_return_pct > 0) stance = "Bullish bias";
  if (predicted_return_pct < 0) stance = "Bearish bias";

  const isBaseline = model_name === "ema_baseline";

  return (
    <div className="panel-body">
      <div className="panel-header">
        <h3>Signal</h3>
        <span className="badge">
          {isBaseline ? "Baseline" : (model_name || "Model")}
        </span>
      </div>

      <div className="metric-row">
        <div className="metric-label">Up Probability</div>
        <div className="metric-value">{upPct}%</div>
      </div>

      <div className="metric-row">
        <div className="metric-label">Predicted Return (1d)</div>
        <div
          className={
            "metric-value " +
            (predicted_return_pct > 0
              ? "pos"
              : predicted_return_pct < 0
              ? "neg"
              : "")
          }
        >
          {retPct}%
        </div>
      </div>

      <div className="metric-row">
        <div className="metric-label">Stance</div>
        <div className="metric-value">
          {stance}
        </div>
      </div>

      {last_trained_at && !isBaseline && (
        <div className="metric-note">
          Trained at: {new Date(last_trained_at).toLocaleString()}
        </div>
      )}

      {isBaseline ? (
        <div className="metric-warning">
          Using EMA baseline / heuristic only. Signals are weak and primarily
          for orientation.
        </div>
      ) : (
        <div className="metric-hint">
          {reliability_hint ||
            "RF on OHLCV-derived features with a neutral band. Research / paper trading only."}
        </div>
      )}

      <div className="metric-note">
        {note ||
          "No guarantee of profitability. Validate on out-of-sample / paper trading."}
      </div>
    </div>
  );
}
