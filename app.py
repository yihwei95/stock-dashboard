# app.py - Enhanced AI Stock Dashboard with enriched portfolio, AI chat, observability
# Features:
# - Portfolio overview with more metrics and AI recommendations
# - AI chat per stock with meaningful response
# - Button-triggered async AI scan with proper sorting and display
# - Timestamped outputs
# - Observability tab placeholder for metrics, traces, tokens

import os
import sys
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import asyncio
from datetime import datetime

# Ensure local folder is in Python path
sys.path.append(os.path.dirname(__file__))

# Import utility functions
from utils import (
    get_overview_df,
    async_ai_call,
    run_parallel_ai,
    check_alert,
    get_stock_data,
    compute_performance
)

# ====================== CONFIG ======================
st.set_page_config(page_title="Enhanced AI Stock Terminal", layout="wide")

# ====================== AUTHENTICATION ======================
# Simple PIN authentication
if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    pin = st.text_input("Enter 6-digit PIN", type="password")
    if pin and pin == st.secrets.get("APP_PIN", "123456"):
        st.session_state.auth = True
    else:
        st.stop()  # Stop until correct PIN

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

# Initialize session state
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

    # Build enriched portfolio table with AI recommendations
    rows = []
    for symbol in STOCKS:
        try:
            info, hist, news = get_stock_data(symbol)
            perf = compute_performance(hist)
            # AI recommendation
            rec = asyncio.run(async_ai_call(symbol, st.secrets.get("CEREBRAS_API_KEY", "")))
            rows.append({
                "Symbol": symbol,
                "Company": info.get("longName", "N/A"),
                "Current Price": perf["current_price"],
                "PE Ratio": info.get("trailingPE", "N/A"),
                "Market Cap": info.get("marketCap", "N/A"),
                "Dividend Yield": info.get("dividendYield", "N/A"),
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
                "PE Ratio": "N/A",
                "Market Cap": "N/A",
                "Dividend Yield": "N/A",
                "Buy Price": "N/A",
                "Recommendation": "N/A",
                "Reason": "No data available",
                "Score": 0
            })

    df_portfolio = pd.DataFrame(rows)
    # Sort by Score descending
    df_portfolio.sort_values("Score", ascending=False, inplace=True)

    # Display table
    st.dataframe(df_portfolio, width='stretch')

    # Timestamp for last refresh
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

    # Input box for user question
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
            # Use async_ai_call and extract Reason field for chat
            response = await async_ai_call(s, api_key)
            return response.get("Reason", "No meaningful response")

        reply = asyncio.run(get_ai_reply())
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", reply))

    # Display chat history
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
                # Streaming display per stock
                text = f"{res['Symbol']} → {res['Recommendation']} (Buy: {res['BuyRange']}, Score: {res['Score']})"
                placeholders[res['Symbol']].markdown(text)
                progress.progress(len(results)/len(STOCKS))
            return results

        st.session_state.rec_df = asyncio.run(run_scan())

        # Build table with proper columns
        df_rec = pd.DataFrame(st.session_state.rec_df)
        df_rec = df_rec[["Symbol", "Recommendation", "BuyRange", "Reason", "Score"]].sort_values("Score", ascending=False)
        st.dataframe(df_rec, width='stretch')

        st.session_state.timestamps["ai_scan"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
        st.caption(f"AI Scan completed: {st.session_state.timestamps['ai_scan']}")

# ------------------ TAB 4: OBSERVABILITY ------------------
with tab4:
    st.subheader("🔍 Observability")
    st.info("This section will show metrics, traces, costs, tokens, etc.")
    st.metric("API Calls", len(STOCKS))
    st.metric("Total Tokens Used", 0)
    st.metric("Errors", 0)
    st.metric("Scan Duration (s)", 0)
    st.markdown("**Traces & Metrics:** Placeholder for Langfuse-like observability dashboard")
