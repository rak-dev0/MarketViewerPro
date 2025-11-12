import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 8000
});

async function request(method, url, params = {}) {
  try {
    const res = await api.request({ method, url, params });
    return { data: res.data, error: null };
  } catch (err) {
    console.error("API error", method, url, err);
    const msg =
      err.response?.data?.detail ||
      err.message ||
      "Unexpected API error";
    return { data: null, error: msg };
  }
}

export const fetchPrice = (ticker, range = "6mo", interval = "1d") =>
  request("get", "/price", { ticker, range, interval });

export const fetchPredict = (ticker) =>
  request("get", "/predict", { ticker });

export const fetchNews = (ticker, limit = 30) =>
  request("get", "/news", { ticker, limit });

export const fetchHealth = () =>
  request("get", "/health");

export const fetchMetrics = () =>
  request("get", "/metrics");
