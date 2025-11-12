import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend
} from "recharts";

function PriceTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;

  const point = payload.reduce((acc, p) => {
    acc[p.dataKey] = p.value;
    return acc;
  }, {});

  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-date">{label}</div>
      <div className="chart-tooltip-row">
        <span>Open</span>
        <span>{point.open != null ? point.open.toFixed(2) : "-"}</span>
      </div>
      <div className="chart-tooltip-row">
        <span>Mid</span>
        <span>{point.mid != null ? point.mid.toFixed(2) : "-"}</span>
      </div>
      <div className="chart-tooltip-row">
        <span>Close</span>
        <span>{point.close != null ? point.close.toFixed(2) : "-"}</span>
      </div>
    </div>
  );
}

export default function PriceChart({ priceData }) {
  if (!priceData) {
    return <div className="panel-body">Loading price data...</div>;
  }

  if (!priceData.candles || priceData.candles.length === 0) {
    return (
      <div className="panel-body">
        <div className="panel-header">
          <h2>{priceData.ticker}</h2>
          <span className="badge">
            {priceData.source?.toUpperCase() || "N/A"}
            {priceData.stale && " • STALE"}
          </span>
        </div>
        <p>No candles returned from API.</p>
      </div>
    );
  }

  const data = priceData.candles.map((c) => {
    const high = Number(c.high);
    const low = Number(c.low);
    const midRaw = (high + low) / 2;
    const mid = Number.isFinite(midRaw) ? midRaw : Number(c.close);

    return {
      time: new Date(c.timestamp).toLocaleDateString(),
      open: Number(c.open),
      close: Number(c.close),
      mid
    };
  });

  const nonZero = data.some(
    (d) => d.open > 0 || d.close > 0 || d.mid > 0
  );

  if (!nonZero) {
    return (
      <div className="panel-body">
        <div className="panel-header">
          <h2>{priceData.ticker}</h2>
          <span className="badge">
            {priceData.source?.toUpperCase() || "N/A"}
            {priceData.stale && " • STALE"}
          </span>
        </div>
        <p>Received only zero/invalid prices. Check upstream data.</p>
      </div>
    );
  }

  return (
    <div className="panel-body">
      <div className="panel-header">
        <h2>{priceData.ticker}</h2>
        <span className="badge">
          {priceData.source?.toUpperCase() || "N/A"}
          {priceData.stale && " • STALE"}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart
          data={data}
          margin={{ top: 10, right: 20, left: 0, bottom: 10 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.18} />
          <XAxis
            dataKey="time"
            minTickGap={20}
            tick={{ fontSize: 10, fill: "#9ca3af" }}
          />
          <YAxis
            domain={["auto", "auto"]}
            tick={{ fontSize: 10, fill: "#9ca3af" }}
          />
          <Tooltip content={<PriceTooltip />} />
          <Legend
            verticalAlign="top"
            align="right"
            iconSize={8}
            wrapperStyle={{ fontSize: 10, color: "#9ca3af" }}
          />
          <Line
            type="monotone"
            dataKey="close"
            name="Close"
            stroke="#38bdf8"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3 }}
          />
          <Line
            type="monotone"
            dataKey="mid"
            name="Mid"
            stroke="#22c55e"
            strokeWidth={1}
            dot={false}
            opacity={0.7}
          />
          <Line
            type="monotone"
            dataKey="open"
            name="Open"
            stroke="#f97316"
            strokeWidth={1}
            dot={false}
            opacity={0.7}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
