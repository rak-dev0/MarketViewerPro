import React from "react";

function sentimentLabel(s) {
  if (s > 0.15) return "Positive";
  if (s < -0.15) return "Negative";
  return "Neutral";
}

export default function NewsList({ items }) {
  if (!items) {
    return (
      <div className="panel-body">
        <h2>News &amp; Sentiment</h2>
        <p>Loading news...</p>
      </div>
    );
  }

  return (
    <div className="panel-body">
      <div className="panel-header">
        <h2>News &amp; Sentiment</h2>
      </div>
      {items.length === 0 && <p>No news found.</p>}
      <ul className="news-list">
        {items.map((n) => (
          <li key={n.url} className="news-item">
            <a href={n.url} target="_blank" rel="noreferrer">
              {n.title}
            </a>
            <div className="news-meta">
              <span>{n.source}</span>
              {n.published_at && (
                <span>
                  {" "}
                  • {new Date(n.published_at).toLocaleString()}
                </span>
              )}
            </div>
            <div className="news-tags">
              <span
                className={
                  "sentiment " +
                  sentimentLabel(n.sentiment).toLowerCase()
                }
              >
                {sentimentLabel(n.sentiment)}
              </span>
              {n.relevance_score > 0.5 && (
                <span className="tag">Relevant</span>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
