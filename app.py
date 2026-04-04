import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import os
import time
from datetime import datetime, timedelta

from utils import *


# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="Stock Dashboard", layout="wide")


# ====================== AUTH ======================
def check_password():
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


if check_password():

    # Init timestamp store
    if "action_timestamps" not in st.session_state:
        st.session_state.action_timestamps = {}

    STOCKS = ["AAPL", "MSFT", "GOOGL"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Portfolio", "Deep Dive", "AI Analysis", "Logs", "Dictionary"
    ])

    # ====================== TAB 1 ======================
    with tab1:
        st.subheader("Portfolio")

        st.caption(f"Last Refresh: {st.session_state.action_timestamps.get('refresh','Never')}")

        if st.button("Refresh Data"):
            progress = st.progress(0)
            status = st.empty()

            status.info("Clearing cache...")
            progress.progress(30)
            st.cache_data.clear()

            status.info("Fetching data...")
            progress.progress(70)
            time.sleep(0.5)

            status.success("Done")
            progress.progress(100)

            st.session_state.action_timestamps["refresh"] = datetime.now().strftime("%d %b %Y %H:%M:%S")
            st.rerun()

        df = get_overview_df(STOCKS)

        st.dataframe(
            df,
            column_config={
                "30-Day Trend": st.column_config.LineChartColumn("Trend")
            },
            use_container_width=True
        )

    # ====================== TAB 2 ======================
    with tab2:
        sel = st.selectbox("Select", STOCKS)

        info, hist, _ = get_stock_data(sel)
        perf = compute_performance(hist)

        fig = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close']
        )])

        st.plotly_chart(fig, use_container_width=True)

        st.metric("Price", perf["current_price"])
        st.metric("Change %", perf["day_change_pct"])

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if p := st.chat_input("Ask AI"):
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.info("Thinking...")

                reply, _ = chat_with_cerebras(p, st.session_state.chat_history, st.secrets.get("CEREBRAS_API_KEY",""), sel)

                placeholder.markdown(reply)

            st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # ====================== TAB 3 ======================
    with tab3:
        st.caption(f"Last AI Scan: {st.session_state.action_timestamps.get('ai','Never')}")

        if st.button("Run AI Scan"):
            progress = st.progress(0)
            status = st.empty()

            results = []

            for i, s in enumerate(STOCKS):
                status.info(f"Analyzing {s}")
                res = get_batch_recommendations([s], st.secrets.get("CEREBRAS_API_KEY",""))
                results.extend(res)
                progress.progress((i+1)/len(STOCKS))

            status.success("Completed")

            st.session_state.rec_df = pd.DataFrame(results)
            st.session_state.action_timestamps["ai"] = datetime.now().strftime("%d %b %Y %H:%M:%S")

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
            st.write("Valuation metric")

        with st.expander("Market Cap"):
            st.write("Company value")