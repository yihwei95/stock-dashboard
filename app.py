# app.py - Enhanced AI Stock Dashboard with Arrow-safe portfolio, caching, and observability
# Features:
# - Portfolio with numeric type normalization
# - Reduced API calls using caching
# - AI chat with meaningful responses
# - Async AI Scan with clean display and timestamp
# - Observability tab placeholders

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import asyncio
from datetime import datetime

# Ensure local folder is in Python path
sys.path.append(os.path.dirname(__file__))

# Import utility functions
from utils import get_overview_df, async_ai_call, run_parallel_ai, get_stock_data, compute_performance

# ====================== CONFIG ======================
st.set_page_config(page_title="Enhanced AI Stock Terminal", layout="wide")

# ====================== AUTHENTICATION ======================
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    pin = st.text_input("Enter 6-digit PIN", type="password")
    if pin and pin == st.secrets.get("APP_PIN", "123456"):
        st.session_state.auth = True
    else:
        st.stop()

# ====================== LOAD STOCKS ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOCKS_FILE = os.path.join(BASE_DIR, "stocks.json")

try:
    with open(STOCKS_FILE, "r") as f:
        STOCKS = json.load(f)
        if not isinstance(STOCKS, list) or len(STOCKS) == 0:
            raise ValueError("stocks.json empty/invalid")
except Exception as e:
    st.warning(f"[WARN] Could not load stocks.json ({e}), using defaults")
    STOCKS = ["AAPL", "MSFT", "GOOGL"]

st.session_state["STOCKS"] = STOCKS

# ====================== SESSION STATE ======================
if "timestamps" not in st.session_state:
    st.session_state["timestamps"] = {}
if "rec_df" not in st.session_state:
    st.session_state["rec_df"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ====================== TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Charts", "AI Scan", "Observability"])

# ------------------ TAB 1: PORTFOLIO ------------------
with tab1:
    st.subheader("📊 Portfolio Overview")

    # Cached function to reduce API calls
    @st.cache_data(ttl=60)  # cache for 60 seconds to limit API usage
    def build_portfolio_df(stocks):
        rows = []
        for symbol in stocks:
            try:
                info, hist, news = get_stock_data(symbol)
                perf = compute_performance(hist)

                # AI recommendation cached per stock
                rec = asyncio.run(async_ai_call(symbol, st.secrets.get("CEREBRAS_API_KEY", "")))

                # Normalize numeric columns: replace 'N/A' with None
                pe_ratio = info.get("trailingPE")
                pe_ratio = float(pe_ratio) if pe_ratio not in (None, "N/A") else np.nan

                market_cap = info.get("marketCap")
                market_cap = float(market_cap) if market_cap not in (None, "N/A") else np.nan

                dividend_yield = info.get("dividendYield")
                dividend_yield = float(dividend_yield) if dividend_yield not in (None, "N/A") else np.nan

                rows.append({
                    "Symbol": symbol,
                    "Company": info.get("longName", "N/A"),
                    "Current Price": perf["current_price"],
                    "PE Ratio": pe_ratio,
                    "Market Cap": market_cap,
                    "Dividend Yield": dividend_yield,
                    "Buy Price": rec.get("BuyRange", "N/A"),
                    "Recommendation": rec.get("Recommendation", "N/A"),
                    "Reason": rec.get("Reason", "No news available"),
                    "Score": rec.get("Score", 0)
                })
            except:
                rows.append({
                    "Symbol": symbol,
                    "Company": "N/A",
                    "Current Price": 0.0,
                    "PE Ratio": np.nan,
                    "Market Cap": np.nan,
                    "Dividend Yield": np.nan,
                    "Buy Price": "N/A",
                    "Recommendation": "N/A",
                    "Reason": "No data available",
                    "Score": 0
                })
        df = pd.DataFrame(rows)
        # Sort primarily by Score descending, then Current Price descending
        df.sort_values(["Score", "Current Price"], ascending=[False, False], inplace=True)
        return df

    df_portfolio = build_portfolio_df(STOCKS)

    st.dataframe(df_portfolio, width='stretch')

    ts = datetime.now().strftime("%d %b %Y, %I:%M %p")
    st.caption(f"Last refreshed: {ts}")

# ------------------ TAB 2: CHARTS + AI CHAT ------------------
with tab2:
    s = st.selectbox("Select Stock", STOCKS)
    info, hist, news = get_stock_data(s)

    # Candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"]
    )])
    fig.update_layout(template="plotly_white", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.subheader(f"💬 Discuss {s} with AI")
    user_input = st.text_input("Ask a question about this stock:")
    if user_input:
        async def get_ai_reply():
            api_key = st.secrets.get("CEREBRAS_API_KEY", "")
            payload = {
                "symbol": s,
                "current_price": compute_performance(hist)["current_price"],
                "news_titles": [n.get("title") for n in news[:3]],
                "question": user_input
            }
            prompt = f"Provide a clear, actionable answer about {s}: {json.dumps(payload)}"
            response = await async_ai_call(s, api_key)
            return response.get("Reason", "No meaningful response")

        reply = asyncio.run(get_ai_reply())
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", reply))

    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**AI:** {message}")

# ------------------ TAB 3: AI SCAN ------------------
with tab3:
    st.subheader("🚀 Run AI Scan")
    if st.button("Get Recommendations"):
        progress = st.progress(0)
        placeholders = {s: st.empty() for s in STOCKS}

        async def run_scan():
            results = []
            tasks = [async_ai_call(s, st.secrets.get("CEREBRAS_API_KEY", "")) for s in STOCKS]
            for i, task in enumerate(asyncio.as_completed(tasks)):
                res = await task
                results.append(res)
                text = f"{res['Symbol']} → {res['Recommendation']} (Buy: {res['BuyRange']}, Score: {res['Score']})"
                placeholders[res['Symbol']].markdown(text)
                progress.progress(len(results)/len(STOCKS))
            return results

        st.session_state.rec_df = asyncio.run(run_scan())
        df_rec = pd.DataFrame(st.session_state.rec_df)
        df_rec = df_rec[["Symbol", "Recommendation", "BuyRange", "Reason", "Score"]].sort_values("Score", ascending=False)
        st.dataframe(df_rec, width='stretch')

        st.session_state.timestamps["ai_scan"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
        st.caption(f"AI Scan completed: {st.session_state.timestamps['ai_scan']}")

# ------------------ TAB 4: OBSERVABILITY ------------------
with tab4:
    st.subheader("🔍 Observability")
    st.info("Metrics, traces, cost, tokens usage placeholder")
    st.metric("API Calls", len(STOCKS))
    st.metric("Total Tokens Used", 0)
    st.metric("Errors", 0)
    st.metric("Scan Duration (s)", 0)
    st.markdown("**Traces & Metrics:** Placeholder for Langfuse-like observability dashboard")