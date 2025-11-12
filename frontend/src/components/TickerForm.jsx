import React, { useState, useEffect } from "react";

const presets = ["1mo", "3mo", "6mo", "1y", "ytd", "max"];

export default function TickerForm({ currentTicker, currentRange, onSubmit }) {
  const [ticker, setTicker] = useState(currentTicker || "AAPL");
  const [range, setRange] = useState(currentRange || "6mo");

  useEffect(() => {
    setTicker(currentTicker || "AAPL");
    setRange(currentRange || "6mo");
  }, [currentTicker, currentRange]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!ticker) return;
    onSubmit(ticker.toUpperCase(), range);
  };

  return (
    <form className="ticker-form" onSubmit={handleSubmit}>
      <input
        className="ticker-input"
        value={ticker}
        onChange={(e) => setTicker(e.target.value)}
        placeholder="Ticker (e.g. AAPL)"
      />
      <div className="range-buttons">
        {presets.map((p) => (
          <button
            key={p}
            type="button"
            className={p === range ? "range-btn active" : "range-btn"}
            onClick={() => setRange(p)}
          >
            {p.toUpperCase()}
          </button>
        ))}
      </div>
      <button className="primary-btn" type="submit">
        Load
      </button>
    </form>
  );
}
