# app.py - Full Streamlit Stock Dashboard with legend + AI chatbox
# Features:
# - Portfolio overview with jargon legend
# - Candlestick charts per stock
# - AI chat discussion per stock
# - Async AI scan with scoring
# - Real-time alerts based on AI buy range

import os
import sys
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import asyncio
from datetime import datetime

# Ensure local folder is in Python path to import utils
sys.path.append(os.path.dirname(__file__))

# Import helper functions from utils.py
from utils import (
    get_overview_df,    # Build main portfolio dataframe
    async_ai_call,      # Async AI recommendation call
    run_parallel_ai,    # Run multiple AI calls concurrently
    check_alert,        # Check if current price triggers alert
    get_stock_data,     # Fetch stock info, history, news
    compute_performance # Compute current price and day change
)

# ====================== CONFIG ======================
# Set page title and wide layout
st.set_page_config(page_title="AI Stock Terminal", layout="wide")

# ====================== AUTHENTICATION ======================
# Simple PIN-based authentication using session_state
if "auth" not in st.session_state:
    st.session_state.auth = False

# Prompt for PIN if not authenticated
if not st.session_state.auth:
    pin = st.text_input("Enter 6-digit PIN", type="password")
    if pin and pin == st.secrets.get("APP_PIN", "123456"):
        st.session_state.auth = True
    else:
        st.stop()  # Stop app until correct PIN is entered

# ====================== LOAD STOCKS ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOCKS_FILE = os.path.join(BASE_DIR, "stocks.json")

# Attempt to load stocks.json; fallback to defaults if fails
try:
    with open(STOCKS_FILE, "r") as f:
        STOCKS = json.load(f)
        if not isinstance(STOCKS, list) or len(STOCKS) == 0:
            raise ValueError("stocks.json empty/invalid")
except Exception as e:
    st.warning(f"[WARN] Could not load stocks.json ({e}), using defaults")
    STOCKS = ["AAPL", "MSFT", "GOOGL"]

# Save stock list to session state
st.session_state["STOCKS"] = STOCKS

# Initialize session state variables if missing
if "timestamps" not in st.session_state:
    st.session_state["timestamps"] = {}
if "rec_df" not in st.session_state:
    st.session_state["rec_df"] = []

# ====================== TABS ======================
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Charts", "AI Scan", "Alerts"])

# ------------------ TAB 1: PORTFOLIO ------------------
with tab1:
    st.subheader("📊 Portfolio Overview")
    
    # Build portfolio dataframe using utils
    df = get_overview_df(STOCKS)
    
    # Display dataframe; 'stretch' width replaces deprecated use_container_width
    st.dataframe(
        df,
        column_config={"30-Day Trend": st.column_config.LineChartColumn("30d Trend")},
        width='stretch'
    )

    # ------------------ Stocks Market Jargon Legend ------------------
    st.markdown("### 📖 Stocks Market Jargon Legend")
    jargon_dict = {
        "Symbol": "Ticker symbol of the stock",
        "Company": "Company full name",
        "Price": "Current market price",
        "Change %": "Percentage change from previous trading day",
        "30-Day Trend": "Closing prices trend over last 30 trading days",
        "BuyRange": "Suggested buying price range by AI",
        "Recommendation": "AI recommendation: Buy / Neutral / Not Buy",
        "Score": "Numeric score for recommendation"
    }
    legend_df = pd.DataFrame(jargon_dict.items(), columns=["Term", "Description"])
    st.table(legend_df)

# ------------------ TAB 2: CHARTS + AI CHAT ------------------
with tab2:
    s = st.selectbox("Select Stock", STOCKS)  # Dropdown to pick stock
    info, hist, _ = get_stock_data(s)         # Fetch stock info and history

    # Candlestick chart of last 1 year
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

    # Initialize chat history in session_state if missing
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input for AI discussion
    user_input = st.text_input("Ask a question about this stock:")
    if user_input:
        async def get_ai_reply():
            api_key = st.secrets.get("CEREBRAS_API_KEY", "")
            # Prepare payload with stock info and user question
            payload = {
                "symbol": s,
                "current_price": compute_performance(hist)["current_price"],
                "question": user_input
            }
            prompt = f"Answer concisely about {s} based on data: {json.dumps(payload)}"
            # Use async_ai_call for demonstration; returns structured JSON
            response = await async_ai_call(s, api_key)
            return response.get("Reason", "No response")

        # Run async call synchronously for Streamlit
        reply = asyncio.run(get_ai_reply())

        # Save chat messages
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
    if st.button("🚀 Run AI Scan (Async + Streaming)"):
        status = st.empty()
        progress = st.progress(0)
        placeholders = {s: st.empty() for s in STOCKS}

        # Async function to run multiple AI calls concurrently
        async def run():
            results = []
            tasks = [async_ai_call(s, st.secrets.get("CEREBRAS_API_KEY", "")) for s in STOCKS]
            for i, task in enumerate(asyncio.as_completed(tasks)):
                res = await task
                results.append(res)
                symbol = res.get("Symbol")
                text = f"{symbol} → {res.get('Recommendation')} (Score {res.get('Score')})"
                # Streaming display of recommendation
                for j in range(len(text)):
                    placeholders[symbol].markdown(text[:j+1])
                    await asyncio.sleep(0.005)
                progress.progress(len(results)/len(STOCKS))
            return results

        st.session_state.rec_df = asyncio.run(run())
        df = pd.DataFrame(st.session_state.rec_df).sort_values("Score", ascending=False)
        st.dataframe(df, width='stretch')
        st.session_state.timestamps["ai"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
        st.success(f"✅ AI Scan completed at {st.session_state.timestamps['ai']}")

# ------------------ TAB 4: ALERTS ------------------
with tab4:
    st.subheader("🔔 Real-time Alerts")
    if not st.session_state.rec_df:
        st.info("Run AI Scan first to generate alerts")
    else:
        alerts = []
        for row in st.session_state.rec_df:
            price = row.get("current_price", 0)
            buy_range = row.get("BuyRange", "0-0")
            if check_alert(price, buy_range):
                alerts.append(row["Symbol"])
        if alerts:
            st.error(f"🔥 ALERT: {alerts}")
        else:
            st.success("No alerts currently")