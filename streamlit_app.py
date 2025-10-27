# =====================================================
# 📊 Stock & Crypto Tracker + AI Analysis Dashboard
# Author: Lyra
# Version: 2.0 (Optimized for Streamlit Cloud)
# =====================================================

import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.graph_objs as go
from datetime import datetime, timedelta
import concurrent.futures
import openai
import os

# -----------------------------
# 🔧 CONFIG
# -----------------------------
st.set_page_config(page_title="Stock & Crypto AI Tracker", layout="wide", page_icon="📈")

st.title("📈 Stock & Crypto Performance Dashboard (AI-Powered)")

STOCKS = ["IONQ", "ENVX", "AMD", "RR"]
CRYPTOS = ["ADA", "XRP"]
YEARS_HISTORY = 5
REFRESH_INTERVAL_MINUTES = 5
COINGECKO_IDS = {"ADA": "cardano", "XRP": "ripple"}

# Optional: store your OpenAI API key as Streamlit Secret
openai.api_key = st.secrets.get("OPENAI_API_KEY", None)

# -----------------------------
# 🧠 AI SUMMARY FUNCTION
# -----------------------------
def ai_summarize(market_data):
    """
    Generate an AI summary based on daily performance and volatility data.
    """
    if not openai.api_key:
        return "⚠️ No API key found. Add your OpenAI key in Streamlit Cloud Secrets."

    text_summary = market_data.to_markdown(index=False)
    prompt = f"""
You are a financial analyst. Analyze today's performance of the following assets:

{text_summary}

Write a concise 3-paragraph report:
1. Overview of market movements and highlights.
2. Compare performance across assets (stocks vs crypto).
3. Mention any unusual volatility or significant changes.

Keep tone: analytical, factual, concise. Avoid financial advice.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a professional market analyst."},
                      {"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ AI summary unavailable: {e}"

# -----------------------------
# 📦 DATA FUNCTIONS
# -----------------------------
@st.cache_data(ttl=REFRESH_INTERVAL_MINUTES * 60)
def get_stock_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=YEARS_HISTORY * 365)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df.reset_index(inplace=True)
    return df

@st.cache_data(ttl=REFRESH_INTERVAL_MINUTES * 60)
def get_crypto_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": YEARS_HISTORY * 365, "interval": "daily"}
    r = requests.get(url, params=params)
    data = r.json()
    prices = pd.DataFrame(data["prices"], columns=["Date", "Price"])
    prices["Date"] = pd.to_datetime(prices["Date"], unit="ms")
    return prices

def quarterly_comparison(df):
    df["Quarter"] = df["Date"].dt.to_period("Q")
    grouped = df.groupby("Quarter")["Close"].mean().reset_index()
    grouped["YoY_Change_%"] = grouped["Close"].pct_change(periods=4) * 100
    return grouped

# -----------------------------
# ⚡ ASYNC FETCH
# -----------------------------
def fetch_all_data():
    stock_data, crypto_data = {}, {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Stocks
        stock_futures = {executor.submit(get_stock_data, s): s for s in STOCKS}
        for f in concurrent.futures.as_completed(stock_futures):
            s = stock_futures[f]
            stock_data[s] = f.result()
        # Crypto
        crypto_futures = {executor.submit(get_crypto_data, COINGECKO_IDS[c]): c for c in CRYPTOS}
        for f in concurrent.futures.as_completed(crypto_futures):
            c = crypto_futures[f]
            crypto_data[c] = f.result()
    return stock_data, crypto_data

# -----------------------------
# 📊 PLOTTING UTILITIES
# -----------------------------
def plot_chart(df, title, y_col="Close"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df[y_col], mode="lines", name=title))
    fig.update_layout(title=title, height=400, xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

def normalize_comparison(data_dict, y_col):
    df_norm = pd.DataFrame()
    for name, df in data_dict.items():
        df_norm[name] = df[y_col] / df[y_col].iloc[0] * 100
    df_norm["Date"] = data_dict[list(data_dict.keys())[0]]["Date"]
    fig = go.Figure()
    for col in df_norm.columns[:-1]:
        fig.add_trace(go.Scatter(x=df_norm["Date"], y=df_norm[col], mode="lines", name=col))
    fig.update_layout(title="Cross-Asset Comparison (Indexed = 100)", xaxis_title="Date", yaxis_title="Indexed Value")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 🚀 MAIN APP
# -----------------------------
st.sidebar.header("Controls")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()

st.sidebar.info("Data auto-refreshes every 5 minutes")

# Fetch all data
with st.spinner("Fetching latest data..."):
    stock_data, crypto_data = fetch_all_data()

# -----------------------------
# OVERVIEW METRICS
# -----------------------------
st.header("📈 Market Overview")

cols = st.columns(len(STOCKS))
for i, ticker in enumerate(STOCKS):
    df = stock_data[ticker]
    last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
    delta = (last - prev) / prev * 100
    cols[i].metric(label=ticker, value=f"${last:,.2f}", delta=f"{delta:.2f}%")

cols = st.columns(len(CRYPTOS))
for i, c in enumerate(CRYPTOS):
    df = crypto_data[c]
    last, prev = df["Price"].iloc[-1], df["Price"].iloc[-2]
    delta = (last - prev) / prev * 100
    cols[i].metric(label=c, value=f"${last:,.4f}", delta=f"{delta:.2f}%")

# -----------------------------
# HISTORICAL CHARTS
# -----------------------------
st.header("📊 Historical Trends")

col1, col2 = st.columns(2)
with col1:
    for t, df in stock_data.items():
        plot_chart(df, f"{t} Stock Price")
with col2:
    for c, df in crypto_data.items():
        plot_chart(df, f"{c} Crypto Price", "Price")

# -----------------------------
# QUARTERLY COMPARISON
# -----------------------------
st.header("📅 Quarterly Comparison (YoY %)")
for t, df in stock_data.items():
    qc = quarterly_comparison(df)
    st.subheader(f"{t}")
    st.dataframe(qc.tail(8))

# -----------------------------
# CROSS-ASSET COMPARISON
# -----------------------------
st.header("⚖️ Like-for-Like Comparison")
stock_norm = {k: v.reset_index() for k, v in stock_data.items()}
normalize_comparison(stock_norm, "Close")

# -----------------------------
# DAILY SUMMARY
# -----------------------------
st.header("🧾 Daily Summary")
summary = []
for t, df in stock_data.items():
    change = df["Close"].pct_change().iloc[-1] * 100
    summary.append({"Asset": t, "Type": "Stock", "Daily % Change": change})
for c, df in crypto_data.items():
    change = df["Price"].pct_change().iloc[-1] * 100
    summary.append({"Asset": c, "Type": "Crypto", "Daily % Change": change})
summary_df = pd.DataFrame(summary)
st.dataframe(summary_df)

# -----------------------------
# ⚠️ ALERTS
# -----------------------------
alerts = summary_df[summary_df["Daily % Change"].abs() > 5]
if not alerts.empty:
    st.warning("⚠️ Significant movements detected (>5%):")
    st.table(alerts)
else:
    st.success("✅ Market stable today.")

# -----------------------------
# 🧠 AI ANALYSIS
# -----------------------------
st.header("🤖 AI Market Summary")
ai_report = ai_summarize(summary_df)
st.markdown(ai_report)
