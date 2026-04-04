# Core UI framework
import streamlit as st

# Data handling
import json
import pandas as pd

# Visualization
import plotly.graph_objects as go

# System utilities
import os
from datetime import datetime, timedelta

# Import business logic
from utils import (
    get_stock_data,
    get_overview_df,
    get_batch_recommendations,
    chat_with_cerebras,
    compute_performance
)


# ====================== PAGE CONFIG ======================

# Configure page layout and branding
st.set_page_config(
    page_title="Secure Stock Dashboard",
    page_icon="📈",
    layout="wide"
)


# ====================== UI STYLING ======================

# Inject custom CSS to enforce light theme
st.markdown("""
    <style>
        .stApp { background-color: #ffffff; color: #111111; }
    </style>
""", unsafe_allow_html=True)


# ====================== AUTHENTICATION ======================

def check_password():
    """
    Simple PIN-based authentication with session expiry.
    """

    def validate_pin():
        # Compare input PIN with stored secret
        if st.session_state["pin_input"] == st.secrets.get("APP_PIN", "123456"):
            st.session_state["authenticated"] = True
            st.session_state["auth_time"] = datetime.now()
        else:
            st.error("❌ Invalid PIN")

    # Initialize authentication state
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    # Check session expiry (12 hours)
    if st.session_state["authenticated"]:
        if datetime.now() - st.session_state["auth_time"] > timedelta(hours=12):
            st.session_state["authenticated"] = False
            st.warning("Session expired.")

    # If not authenticated → block app
    if not st.session_state["authenticated"]:
        st.title("🔒 Access Restricted")

        st.text_input(
            "Enter PIN",
            type="password",
            key="pin_input",
            on_change=validate_pin
        )

        st.stop()

    return True


# ====================== MAIN APP ======================

if check_password():

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Load API key securely
    if "api_key" not in st.session_state:
        st.session_state.api_key = st.secrets.get("CEREBRAS_API_KEY", "")

    # Load stock list from file
    if os.path.exists("stocks.json"):
        try:
            with open("stocks.json", "r") as f:
                STOCKS = json.loads(f.read().strip())
        except:
            STOCKS = ["AAPL", "MSFT", "GOOGL"]
    else:
        STOCKS = ["AAPL", "MSFT", "GOOGL"]

    # Track last refresh timestamp
    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = datetime.now().strftime("%d %b %Y, %I:%M %p")

    # App title
    st.title("📈 Professional Stock Dashboard")

    # ====================== SIDEBAR ======================

    with st.sidebar:
        st.success("🔐 Session Active")

        # Logout button
        if st.button("Log Out"):
            st.session_state["authenticated"] = False
            st.rerun()

    # ====================== TABS ======================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio",
        "🔍 Deep Dive",
        "🤖 AI Analysis",
        "📈 Logs",
        "📖 Dictionary"
    ])

    # ====================== TAB 1 ======================

    with tab1:

        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        # Load portfolio data
        df = get_overview_df(STOCKS)

        # Display table
        st.dataframe(df, use_container_width=True)

    # ====================== TAB 2 ======================

    with tab2:

        # Select stock
        sel = st.selectbox("Select Ticker", STOCKS)

        # Fetch data
        info, hist, news = get_stock_data(sel)
        perf = compute_performance(hist)

        # Candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close']
            )
        ])

        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        st.metric("Price", perf["current_price"])
        st.metric("Change %", perf["day_change_pct"])

        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display messages
        for m in st.session_state.chat_history:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # Chat input
        if prompt := st.chat_input("Ask AI..."):

            # Store user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Get AI response
            reply, _ = chat_with_cerebras(
                prompt,
                st.session_state.chat_history,
                st.session_state.api_key,
                sel
            )

            # Store response
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

            st.rerun()

    # ====================== TAB 3 ======================

    with tab3:

        # Run AI scan
        if st.button("Run AI Scan"):

            results = []

            for s in STOCKS:
                res = get_batch_recommendations([s], st.session_state.api_key)
                results.extend(res)

            # Store results
            st.session_state.rec_df = pd.DataFrame(results)

        # Display results
        if "rec_df" in st.session_state:
            st.dataframe(st.session_state.rec_df)

    # ====================== TAB 4 ======================

    with tab4:

        # Display logs if available
        if "llm_logs" in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state.llm_logs))

    # ====================== TAB 5 ======================

    with tab5:

        st.header("📖 Dictionary")

        st.write("Basic stock market terms explained.")