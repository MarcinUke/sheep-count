# =====================================================
# Stock & Crypto Tracker + AI Analysis Dashboard
# Optimized for Streamlit Cloud
# =====================================================

from __future__ import annotations
import os
import time
import math
import json
import concurrent.futures
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objs as go
import streamlit as st

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Stock & Crypto AI Tracker", layout="wide")

# -----------------------------
# App Header / Controls
# -----------------------------
st.title("📈 Stock & Crypto Performance Dashboard (AI-Powered)")
st.sidebar.header("Controls")

# Default configuration (you can edit these)
YEARS_HISTORY = 5
REFRESH_INTERVAL_MINUTES = 5
# User tickers (you asked for these)
USER_STOCKS = ["IONQ", "ENVX", "AMD", "RR"]  # We'll map RR -> RR.L below
USER_CRYPTOS = ["ADA", "XRP"]

# Map user-friendly tickers to data-source tickers where needed
# Rolls-Royce on LSE is "RR.L" in Yahoo Finance
YF_TICKER_MAP = {"RR": "RR.L"}  # if not present, use ticker as-is

# CoinGecko IDs
COINGECKO_IDS = {"ADA": "cardano", "XRP": "ripple"}

# -----------------------------
# OpenAI setup (supports both old and new SDKs)
# -----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = None
using_new_openai = False
if OPENAI_API_KEY:
    try:
        # New SDK style
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=OPENAI_API_KEY)
        using_new_openai = True
    except Exception:
        try:
            # Legacy SDK style
            import openai  # type: ignore
            openai.api_key = OPENAI_API_KEY
            client = openai
            using_new_openai = False
        except Exception:
            client = None

# -----------------------------
# Helper: resilient HTTP GET with retries/backoff
# -----------------------------
def fetch_json(url: str, params: dict | None = None, retries: int = 3, timeout: int = 12, backoff: float = 1.6):
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            # brief exponential backoff
            time.sleep((backoff ** attempt) + (0.2 * attempt))
    raise last_err if last_err else RuntimeError("Unknown HTTP error")

# -----------------------------
# Data Functions (cached)
# -----------------------------
@st.cache_data(ttl=REFRESH_INTERVAL_MINUTES * 60, show_spinner=False)
def get_stock_data(ticker: str) -> pd.DataFrame:
    """Fetch daily OHLCV for a stock via yfinance, limited to YEARS_HISTORY.
       Handles unnamed datetime index -> 'index' -> 'Date' renaming robustly.
    """
    yf_symbol = YF_TICKER_MAP.get(ticker, ticker)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=YEARS_HISTORY * 365)

    df = yf.download(
        yf_symbol,
        start=start_date,
        end=end_date,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )

    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])

    # Reset index and make sure the date column is named 'Date'
    df = df.reset_index()
    if "Date" not in df.columns:
        # yfinance sometimes uses 'index' if the DatetimeIndex had no name
        if "index" in df.columns:
            df.rename(columns={"index": "Date"}, inplace=True)
        else:
            # last resort: take the first datetime-like column
            first_col = df.columns[0]
            df.rename(columns={first_col: "Date"}, inplace=True)

    # Ensure datetime dtype
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # Some providers omit 'Adj Close' for certain tickers/intervals
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    # Ensure required columns exist even if empty
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

# -----------------------------
# Fetch Everything (parallel + safe)
# -----------------------------
def fetch_all_data(stocks: list[str], cryptos: list[str]):
    stock_data, crypto_data = {}, {}

    max_workers = min(8, len(stocks) + len(cryptos)) or 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        stock_futures = {executor.submit(get_stock_data, s): s for s in stocks}
        crypto_futures = {executor.submit(get_crypto_data, COINGECKO_IDS[c]): c for c in cryptos}

        # Stocks
        for f in concurrent.futures.as_completed(stock_futures):
            sym = stock_futures[f]
            try:
                stock_data[sym] = f.result()
            except Exception as e:
                st.error(f"Stock fetch error for {sym}: {e}")
                stock_data[sym] = pd.DataFrame()

        # Cryptos
        for f in concurrent.futures.as_completed(crypto_futures):
            sym = crypto_futures[f]
            try:
                crypto_data[sym] = f.result()
            except Exception as e:
                st.error(f"Crypto fetch error for {sym}: {e}")
                crypto_data[sym] = pd.DataFrame()

    return stock_data, crypto_data

# -----------------------------
# Plotting Utilities
# -----------------------------
def plot_line(df: pd.DataFrame, title: str, y_col: str, y_label: str = "Price"):
    if df is None or df.empty or y_col not in df.columns:
        st.info(f"No data to plot for {title}.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df[y_col], mode="lines", name=title))
    fig.update_layout(title=title, height=380, xaxis_title="Date", yaxis_title=y_label)
    st.plotly_chart(fig, use_container_width=True)

def plot_candlestick(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        st.info(f"No data to plot for {title}.")
        return
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=title
    )])
    fig.update_layout(title=title, height=420, xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

def normalize_like_for_like(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Normalize multiple price series to a common index (100) on their first overlapping date.
    Aligns by inner join on the intersection of dates.
    """
    frames = []
    for name, ser in series_map.items():
        s = ser.dropna().copy()
        s.name = name
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=1, join="inner").dropna(how="all")
    if df.empty:
        return df
    # rebase each series to 100 on the first row
    base = df.iloc[0]
    df = df.divide(base) * 100.0
    df["Date"] = df.index
    return df.reset_index(drop=True)

def plot_normalized(df_norm: pd.DataFrame, title: str):
    if df_norm is None or df_norm.empty:
        st.info("No overlapping data to compare.")
        return
    fig = go.Figure()
    date_col = "Date"
    for col in df_norm.columns:
        if col == date_col:
            continue
        fig.add_trace(go.Scatter(x=df_norm[date_col], y=df_norm[col], mode="lines", name=col))
    fig.update_layout(title=title, height=420, xaxis_title="Date", yaxis_title="Indexed (100 = start)")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# AI Summary
# -----------------------------
def ai_summarize(summary_df: pd.DataFrame) -> str:
    if client is None or not OPENAI_API_KEY:
        return "⚠️ AI summary disabled: no OpenAI API key found (add OPENAI_API_KEY in Streamlit Secrets)."

    table_md = summary_df.to_markdown(index=False)

    prompt = (
        "You are a professional markets analyst. Analyze today's performance for the assets below.\n\n"
        f"{table_md}\n\n"
        "Write a concise 3-paragraph report:\n"
        "1) Overview of the day's movements and highlights.\n"
        "2) Compare performance across assets (stocks vs. crypto), note dispersion and notable outliers.\n"
        "3) Mention any unusual volatility or significant changes; keep it factual and avoid advice.\n"
        "Tone: analytical, concise (150–220 words). No emojis. No recommendations."
    )

    try:
        if using_new_openai:
            # New SDK
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional market analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=350,
            )
            return resp.choices[0].message.content.strip()
        else:
            # Legacy SDK
            resp = client.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional market analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=350,
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI summary unavailable: {e}"

# -----------------------------
# Sidebar actions
# -----------------------------
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()

st.sidebar.info(f"Data auto-refreshes every {REFRESH_INTERVAL_MINUTES} minutes.")

# -----------------------------
# Fetch Data
# -----------------------------
with st.spinner("Fetching latest market data..."):
    stocks = USER_STOCKS
    cryptos = USER_CRYPTOS
    stock_data, crypto_data = fetch_all_data(stocks, cryptos)

# -----------------------------
# Overview Metrics
# -----------------------------
st.header("Market Overview")

# Stocks metrics
stock_cols = st.columns(len(stocks))
for i, t in enumerate(stocks):
    df = stock_data.get(t, pd.DataFrame())
    if df.empty:
        stock_cols[i].metric(label=t, value="N/A", delta="N/A")
        continue
    last = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2] if len(df) > 1 else last
    delta = (last - prev) / prev * 100.0 if prev else 0.0
    stock_cols[i].metric(label=t, value=f"${last:,.2f}", delta=f"{delta:.2f}%")

# Crypto metrics
crypto_cols = st.columns(len(cryptos))
for i, c in enumerate(cryptos):
    df = crypto_data.get(c, pd.DataFrame())
    if df.empty:
        crypto_cols[i].metric(label=c, value="N/A", delta="N/A")
        continue
    last = df["Price"].iloc[-1]
    prev = df["Price"].iloc[-2] if len(df) > 1 else last
    delta = (last - prev) / prev * 100.0 if prev else 0.0
    crypto_cols[i].metric(label=c, value=f"${last:,.4f}", delta=f"{delta:.2f}%")

# -----------------------------
# Charts
# -----------------------------
st.header("Historical Trends")

left, right = st.columns(2)

with left:
    for t, df in stock_data.items():
        if df.empty:
            st.info(f"{t}: no data.")
            continue
        # Candlestick for stocks
        plot_candlestick(df, f"{t} — Daily Candlestick")
        # Optional simple line on Adj Close
        plot_line(df, f"{t} — Adj Close (line)", y_col="Adj Close", y_label="Adj Close")

with right:
    for c, df in crypto_data.items():
        if df.empty:
            st.info(f"{c}: no data.")
            continue
        plot_line(df, f"{c} — USD Price (line)", y_col="Price", y_label="USD Price")

# -----------------------------
# Quarterly Comparison (Stocks)
# -----------------------------
st.header("Quarter-on-Quarter Comparison (Stocks)")
for t, df in stock_data.items():
    qc = quarterly_comparison_stocks(df)
    st.subheader(t)
    if qc.empty:
        st.info("No quarterly data.")
    else:
        st.dataframe(qc.tail(8), use_container_width=True)

# -----------------------------
# Cross-Asset Like-for-Like (Indexed 100)
# -----------------------------
st.header("Like-for-Like Comparison (Indexed to 100)")

# Prepare aligned, normalized series
series_map = {}
for t, df in stock_data.items():
    if not df.empty:
        s = df.set_index("Date")["Close"].astype(float)
        series_map[t] = s
for c, df in crypto_data.items():
    if not df.empty:
        s = df.set_index("Date")["Price"].astype(float)
        series_map[c] = s

df_norm = normalize_like_for_like(series_map)
plot_normalized(df_norm, "Cross-Asset Comparison (Rebased to 100 at First Overlap)")

# -----------------------------
# Daily Summary + Alerts
# -----------------------------
# -----------------------------
# Daily Summary + Alerts
# -----------------------------
st.header("Daily Summary")
summary_rows = []

for t, df in stock_data.items():
    if not df.empty and "Close" in df.columns:
        daily_return = (df["Close"].pct_change().iloc[-1] * 100.0) if len(df) > 1 else 0.0
        summary_rows.append({"Asset": t, "Type": "Stock", "Daily % Change": float(daily_return)})

for c, df in crypto_data.items():
    if not df.empty and "Price" in df.columns:
        daily_return = (df["Price"].pct_change().iloc[-1] * 100.0) if len(df) > 1 else 0.0
        summary_rows.append({"Asset": c, "Type": "Crypto", "Daily % Change": float(daily_return)})

summary_df = pd.DataFrame(summary_rows, columns=["Asset", "Type", "Daily % Change"])

if summary_df.empty:
    st.info("No data available yet for a daily summary.")
else:
    st.dataframe(summary_df, use_container_width=True)

    # Alerts only if the column exists and DataFrame not empty
    if "Daily % Change" in summary_df.columns and not summary_df.empty:
        alerts = summary_df.loc[summary_df["Daily % Change"].abs() > 5.0].copy()
    else:
        alerts = pd.DataFrame(columns=summary_df.columns)

    if not alerts.empty:
        st.warning("Significant movements detected (>5%):")
        st.table(alerts)
    else:
        st.success("No major movements today.")


# -----------------------------
# AI Analysis
# -----------------------------
st.header("AI Market Summary")
st.markdown(ai_summarize(summary_df))
