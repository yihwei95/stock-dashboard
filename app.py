# Import Streamlit UI framework
import streamlit as st

# JSON handling
import json

# Data handling
import pandas as pd

# Plotting
import plotly.graph_objects as go

# OS operations
import os

# Time for smooth UX delays
import time

# Datetime utilities
from datetime import datetime, timedelta

# Parallel execution
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import backend functions
from utils import *


# ====================== PAGE CONFIG ======================

st.set_page_config(
    page_title="Stock Dashboard",
    layout="wide"
)


# ====================== AUTH ======================

def check_password():
    """Simple PIN authentication."""

    def validate():
        if st.session_state["pin"] == st.secrets.get("APP_PIN", "123456"):
            st.session_state.auth = True
            st.session_state.auth_time = datetime.now()
        else:
            st.error("Invalid PIN")

    if "auth" not in st.session_state:
        st.session_state.auth = False

    if st.session_state.auth:
        if datetime.now() - st.session_state.auth_time > timedelta(hours=12):
            st.session_state.auth = False

    if not st.session_state.auth:
        st.text_input("Enter PIN", type="password", key="pin", on_change=validate)
        st.stop()

    return True


# ====================== MAIN APP ======================

if check_password():

    # Initialize timestamps
    if "action_timestamps" not in st.session_state:
        st.session_state.action_timestamps = {}

    # API key
    api_key = st.secrets.get("CEREBRAS_API_KEY", "")

    # Stock list
    STOCKS = ["AAPL", "MSFT", "GOOGL"]

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Portfolio", "Deep Dive", "AI Analysis", "Logs", "Dictionary"
    ])


    # ====================== TAB 1 ======================
    with tab1:
        st.subheader("Portfolio")

        # Show last refresh
        st.caption(f"Last Refresh: {st.session_state.action_timestamps.get('refresh','Never')}")

        # Refresh button
        if st.button("Refresh Data"):
            progress = st.progress(0)
            status = st.empty()

            status.info("Clearing cache...")
            progress.progress(30)
            st.cache_data.clear()

            status.info("Fetching new data...")
            progress.progress(70)
            time.sleep(0.5)

            status.success("Done")
            progress.progress(100)

            st.session_state.action_timestamps["refresh"] = datetime.now().strftime("%d %b %Y %H:%M:%S")
            st.rerun()

        # Load dataframe
        df = get_overview_df(STOCKS)

        # Display table with sparkline
        st.dataframe(
            df,
            column_config={
                "30-Day Trend": st.column_config.LineChartColumn("Trend")
            },
            use_container_width=True
        )


    # ====================== TAB 2 ======================
    with tab2:
        sel = st.selectbox("Select Stock", STOCKS)

        info, hist, _ = get_stock_data(sel)
        perf = compute_performance(hist)

        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"]
        )])

        st.plotly_chart(fig, use_container_width=True)

        st.metric("Price", perf["current_price"])
        st.metric("Change %", perf["day_change_pct"])

        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # Chat input
        if p := st.chat_input("Ask AI"):
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.info("Thinking...")

                reply, _ = chat_with_cerebras(p, st.session_state.chat_history, api_key, sel)

                placeholder.markdown(reply)

            st.session_state.chat_history.append({"role": "assistant", "content": reply})


    # ====================== TAB 3 ======================
    with tab3:
        st.subheader("🚀 AI Scan (Parallel + Live)")

        st.caption(f"Last AI Scan: {st.session_state.action_timestamps.get('ai','Never')}")

        if st.button("Run AI Scan", type="primary"):

            placeholders = {}
            results = []

            # Create placeholders for streaming
            for s in STOCKS:
                placeholders[s] = st.empty()

            progress = st.progress(0)
            total = len(STOCKS)
            done = 0

            # Parallel execution
            with ThreadPoolExecutor(max_workers=5) as executor:

                futures = {
                    executor.submit(get_single_recommendation, s, api_key): s
                    for s in STOCKS
                }

                # Process completed futures
                for future in as_completed(futures):
                    symbol = futures[future]

                    try:
                        res = future.result()

                        # Stream result immediately
                        placeholders[symbol].success(
                            f"**{symbol}** → {res.get('Recommendation')}\n\n{res.get('Reason')}"
                        )

                        results.append(res)

                    except Exception as e:
                        placeholders[symbol].error(f"{symbol} failed: {e}")

                    done += 1
                    progress.progress(done / total)

            st.success("All completed")

            st.session_state.rec_df = pd.DataFrame(results)
            st.session_state.action_timestamps["ai"] = datetime.now().strftime("%d %b %Y %H:%M:%S")

        # Show final table
        if "rec_df" in st.session_state:
            st.dataframe(st.session_state.rec_df)


    # ====================== TAB 4 ======================
    with tab4:
        if "llm_logs" in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state.llm_logs))


    # ====================== TAB 5 ======================
    with tab5:
        st.header("Dictionary")

        with st.expander("P/E Ratio"):
            st.write("Price-to-earnings ratio.")

        with st.expander("Market Cap"):
            st.write("Total company value.")