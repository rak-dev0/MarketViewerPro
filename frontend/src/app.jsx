import React, { useState, useEffect } from "react";
import {
  fetchPrice,
  fetchPredict,
  fetchNews,
  fetchHealth,
  fetchMetrics
} from "./api";
import TickerForm from "./components/TickerForm.jsx";
import PriceChart from "./components/PriceChart.jsx";
import Metrics from "./components/Metrics.jsx";
import NewsList from "./components/NewsList.jsx";
import StatusBar from "./components/StatusBar.jsx";

export default function App() {
  const [ticker, setTicker] = useState("AAPL");
  const [range, setRange] = useState("6mo");
  const [priceData, setPriceData] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [news, setNews] = useState([]);
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadAll = async (t, r) => {
    setLoading(true);
    setError(null);

    try {
      const [priceRes, predRes, newsRes] = await Promise.all([
        fetchPrice(t, r, "1d"),
        fetchPredict(t),
        fetchNews(t, 30)
      ]);

      if (priceRes.error) {
        setError(priceRes.error);
      }

      setPriceData(priceRes.data || null);
      setPrediction(predRes.data || null);
      setNews(newsRes.data?.items || []);
    } catch (e) {
      console.error("loadAll error", e);
      setError("Failed to load data.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    (async () => {
      try {
        const [h, m] = await Promise.all([fetchHealth(), fetchMetrics()]);
        setHealth(h.data || null);
        setMetrics(m.data || null);
      } catch (e) {
        console.error("health/metrics error", e);
      }
      loadAll(ticker, range);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSubmit = (t, r) => {
    setTicker(t);
    setRange(r);
    loadAll(t, r);
  };

  return (
    <div className="app-root">
      <header className="app-header">
        <h1>MarketViewerPro</h1>
        <p className="subtitle">
          Prices, signals & curated news in one compact view.
        </p>
      </header>

      <StatusBar
        health={health}
        priceData={priceData}
        loading={loading}
        error={error}
        metrics={metrics}
      />

      <TickerForm
        currentTicker={ticker}
        currentRange={range}
        onSubmit={handleSubmit}
      />

      <main className="grid">
        <section className="panel wide">
          <PriceChart priceData={priceData} />
        </section>
        <section className="panel">
          <Metrics prediction={prediction} />
        </section>
        <section className="panel tall">
          <NewsList items={news} />
        </section>
      </main>
    </div>
  );
}
