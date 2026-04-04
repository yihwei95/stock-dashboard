"""
app.py - Main Streamlit Stock Dashboard
Features:
- Portfolio overview
- Candlestick charts
- Async AI tactical scan (parallel + streaming)
- Signal scoring + ranking
- Real-time alerts
- Robust stocks.json handling
"""

import os
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import asyncio
from datetime import datetime, timedelta

from utils import get_overview_df, async_ai_call, run_parallel_ai, check_alert

# ====================== CONFIG ======================

st.set_page_config(page_title="AI Stock Terminal", layout="wide")


# ====================== AUTHENTICATION ======================

def check_password():
    """Simple PIN-based authentication for the dashboard."""
    if "auth" not in st.session_state:
        st.session_state.auth = False

    if not st.session_state.auth:
        pin = st.text_input("Enter 6-digit PIN", type="password")
        if pin == st.secrets.get("APP_PIN", "123456"):
            st.session_state.auth = True
            st.experimental_rerun()
        st.stop()

    return True


if check_password():

    # ====================== LOAD STOCKS ======================

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    STOCKS_FILE = os.path.join(BASE_DIR, "stocks.json")

    try:
        with open(STOCKS_FILE, "r") as f:
            STOCKS = json.load(f)
            if not isinstance(STOCKS, list) or len(STOCKS) == 0:
                raise ValueError("stocks.json empty/invalid")
    except Exception as e:
        print(f"[WARN] Could not load stocks.json ({e}), using default")
        STOCKS = ["AAPL", "MSFT", "GOOGL"]

    st.session_state["STOCKS"] = STOCKS

    # Initialize timestamps
    if "timestamps" not in st.session_state:
        st.session_state["timestamps"] = {}

    if "rec_df" not in st.session_state:
        st.session_state["rec_df"] = []

    # ====================== TABS ======================

    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Charts", "AI Scan", "Alerts"])

    # ------------------ TAB 1: PORTFOLIO ------------------
    with tab1:
        st.subheader("📊 Portfolio Overview")
        df = get_overview_df(STOCKS)
        st.dataframe(
            df,
            column_config={
                "30-Day Trend": st.column_config.LineChartColumn("30d Trend")
            },
            use_container_width=True
        )

    # ------------------ TAB 2: CHARTS ------------------
    with tab2:
        s = st.selectbox("Select Stock", STOCKS)
        info, hist, _ = st.session_state.get("data", (None, None, None))  # placeholder
        info, hist, _ = async_ai_call.get_stock_data(s) if hasattr(async_ai_call, "get_stock_data") else (None, None, None)
        if hist is None:
            import yfinance as yf
            ticker = yf.Ticker(s)
            hist = ticker.history(period="1y")

        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"]
        )])
        fig.update_layout(template="plotly_white", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ------------------ TAB 3: AI SCAN ------------------
    with tab3:
        if st.button("🚀 Run AI Scan (Async + Streaming)"):
            status = st.empty()
            progress = st.progress(0)
            placeholders = {s: st.empty() for s in STOCKS}

            async def run():
                results = []
                tasks = [async_ai_call(s, st.secrets.get("CEREBRAS_API_KEY","")) for s in STOCKS]

                for i, task in enumerate(asyncio.as_completed(tasks)):
                    res = await task
                    results.append(res)
                    symbol = res.get("Symbol")
                    text = f"{symbol} → {res.get('Recommendation')} (Score {res.get('Score')})"

                    # Streaming effect per stock
                    for j in range(len(text)):
                        placeholders[symbol].markdown(text[:j+1])
                        await asyncio.sleep(0.005)

                    progress.progress(len(results)/len(STOCKS))

                return results

            # Run async
            st.session_state.rec_df = asyncio.run(run())

            # Sort by score descending
            df = pd.DataFrame(st.session_state.rec_df).sort_values("Score", ascending=False)
            st.dataframe(df)

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