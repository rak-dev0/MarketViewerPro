import React from "react";

export default function StatusBar({
  health,
  priceData,
  loading,
  error,
  metrics
}) {
  const healthLabel = health?.status || "unknown";
  const modelName = metrics?.model?.model_name || "ema_baseline";

  return (
    <div className="status-bar">
      <span>API: {healthLabel}</span>
      <span>Model: {modelName}</span>
      {priceData?.stale && <span>• Using cached data</span>}
      {loading && <span>• Loading...</span>}
      {error && <span className="error">• {error}</span>}
    </div>
  );
}
