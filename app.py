"""
app.py
Header: Main entrypoint with a PIN-access security layer and 12-hour session persistence.
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import os
import time
from datetime import datetime, timedelta

from utils import (
    get_stock_data, get_overview_df, get_batch_recommendations,
    chat_with_cerebras, compute_performance
)

# 1. Page Config
st.set_page_config(page_title="Secure Stock Dashboard", page_icon="📈", layout="wide")

# 2. Force Light Theme CSS
st.markdown("""
    <style>
        .stApp { background-color: #ffffff; color: #111111; }
        [data-testid="stHeader"] { background: rgba(255,255,255,0); }
        .stButton button { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# 3. Security & Session Logic
def check_password():
    """Returns True if the user has a valid PIN session within the last 12 hours."""
    
    def validate_pin():
        if st.session_state["pin_input"] == st.secrets.get("APP_PIN", "1234"):
            st.session_state["authenticated"] = True
            st.session_state["auth_time"] = datetime.now()
            st.success("Access Granted!")
            time.sleep(1)
        else:
            st.error("❌ Invalid PIN")

    # Initialize state
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    # Check if 12 hours have passed since last authentication
    if st.session_state["authenticated"]:
        elapsed_time = datetime.now() - st.session_state["auth_time"]
        if elapsed_time > timedelta(hours=12):
            st.session_state["authenticated"] = False
            st.warning("Session expired. Please re-enter PIN.")

    # Show Login Screen if not authenticated
    if not st.session_state["authenticated"]:
        st.title("🔒 Access Restricted")
        st.write("Please enter the authorization code to access the terminal.")
        st.text_input("Enter 4-6 Digit PIN", type="password", key="pin_input", on_change=validate_pin)
        st.info("Authorized sessions last for 12 hours.")
        st.stop() # Stop execution here until authenticated
        return False
    
    return True

# 4. Main App Execution
if check_password():
    # --- Background Data Prep ---
    os.makedirs("data", exist_ok=True)
    
    if "api_key" not in st.session_state:
        st.session_state.api_key = st.secrets.get("CEREBRAS_API_KEY", "")

    # Load stocks.json
    if os.path.exists("stocks.json"):
        try:
            with open("stocks.json", "r") as f:
                content = f.read().strip()
                STOCKS = json.loads(content) if content else ["AAPL", "MSFT"]
        except:
            STOCKS = ["AAPL", "MSFT"]
    else:
        STOCKS = ["AAPL", "MSFT"]

    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = datetime.now().strftime("%d %B %Y, %I:%M:%S %p")

    # --- UI Layout ---
    st.title("📈 Professional Stock Dashboard")
    
    # Sidebar is now just for Logout and Info
    with st.sidebar:
        st.success("🔐 Authenticated")
        if st.button("Log Out"):
            st.session_state["authenticated"] = False
            st.rerun()
        st.divider()
        st.caption("API connection: Secure (Cerebras)")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio", "🔍 Deep Dive", "🤖 AI Analysis", "📈 Logs"])

    # --- Tab 1: Portfolio ---
    with tab1:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.subheader("Market Overview")
            st.caption(f"⏱️ Last Update: {st.session_state.last_refresh_time}")
        with c2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.session_state.last_refresh_time = datetime.now().strftime("%d %B %Y, %I:%M:%S %p")
                st.rerun()
        
        df = get_overview_df(STOCKS)
        st.dataframe(
            df,
            column_config={
                "Company": st.column_config.TextColumn("Company", width="large"),
                "30-Day Trend": st.column_config.LineChartColumn("Trend", y_min=0),
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "Change %": st.column_config.NumberColumn("Chg %", format="%.2f%%"),
            },
            use_container_width=True, hide_index=True
        )

    # --- Tab 2: Deep Dive ---
    with tab2:
        selected_stock = st.selectbox("Pick a Stock", STOCKS)
        info, hist, news = get_stock_data(selected_stock)
        perf = compute_performance(hist)

        col_chart, col_stats = st.columns([3, 1])
        with col_chart:
            if not hist.empty:
                fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"])])
                fig.update_layout(template="plotly_white", margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
        with col_stats:
            st.metric("Current", f"${perf['current_price']}", f"{perf['day_change_pct']}%")
            st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B")

        st.divider()
        if prompt := st.chat_input(f"Question about {selected_stock}?"):
            if "chat_history" not in st.session_state: st.session_state.chat_history = []
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            reply, _ = chat_with_cerebras(prompt, st.session_state.chat_history, st.session_state.api_key, selected_stock)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    # --- Tab 3: AI Recommendations ---
    with tab3:
        if st.button("🚀 Analyze All Stocks", type="primary", use_container_width=True):
            all_results = []
            p_bar = st.progress(0)
            p_text = st.empty()
            
            for i, symbol in enumerate(STOCKS):
                p_text.markdown(f"**Processing:** {symbol} ({i+1}/{len(STOCKS)})")
                res = get_batch_recommendations([symbol], st.session_state.api_key)
                all_results.extend(res)
                p_bar.progress((i + 1) / len(STOCKS))
            
            p_text.success("Complete!")
            st.session_state.recommendations_df = pd.DataFrame(all_results)
            st.rerun()

        if "recommendations_df" in st.session_state:
            st.table(st.session_state.recommendations_df)

    # --- Tab 4: Observability ---
    with tab4:
        if "llm_logs" in st.session_state:
            st.write(pd.DataFrame(st.session_state.llm_logs))