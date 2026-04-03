"""
app.py
Header: The main Streamlit entrypoint. Handles UI layout, state initialization, 
secrets management, and rendering data visualization components.
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import os
import time
from datetime import datetime

from utils import (
    get_stock_data, get_overview_df, get_batch_recommendations,
    chat_with_cerebras, compute_performance
)

# Set up page layout
st.set_page_config(page_title="Pro Stock Dashboard", page_icon="📈", layout="wide")

# Force Light Theme styling directly via CSS
st.markdown("""
    <style>
        .stApp { background-color: #ffffff; color: #111111; }
        [data-testid="stHeader"] { background: rgba(255,255,255,0); }
    </style>
""", unsafe_allow_html=True)

os.makedirs("data", exist_ok=True)

def load_persistent_data():
    """Loads previous session data from local disk (if available)."""
    if os.path.exists("data/recommendations.json"):
        try:
            st.session_state.recommendations_df = pd.read_json("data/recommendations.json")
        except:
            st.session_state.recommendations_df = pd.DataFrame()
    if "chat_history" not in st.session_state and os.path.exists("data/chat_history.json"):
        try:
            with open("data/chat_history.json", "r") as f:
                st.session_state.chat_history = json.load(f)
        except:
            st.session_state.chat_history = []

def save_persistent_data():
    """Saves current state to local disk for persistence across reruns."""
    if "recommendations_df" in st.session_state:
        st.session_state.recommendations_df.to_json("data/recommendations.json", orient="records")
    if "chat_history" in st.session_state:
        with open("data/chat_history.json", "w") as f:
            json.dump(st.session_state.chat_history, f)

# Initialize data
load_persistent_data()

# Timestamp tracker in Session State
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = datetime.now().strftime("%d %B %Y, %I:%M:%S %p")

st.title("📈 Professional Stock Dashboard")

# ====================== SIDEBAR ======================
st.sidebar.title("⚙️ Settings")

# Check if API key needs to be initialized from secrets
if "api_key" not in st.session_state:
    if "CEREBRAS_API_KEY" in st.secrets:
        st.session_state.api_key = st.secrets["CEREBRAS_API_KEY"]
    else:
        st.session_state.api_key = ""

api_key = st.sidebar.text_input("Cerebras API Key", value=st.session_state.api_key, type="password")
if api_key: st.session_state.api_key = api_key

# Securely load the stock list from stocks.json
if os.path.exists("stocks.json"):
    try:
        with open("stocks.json", "r") as f:
            content = f.read().strip()
            if not content:
                STOCKS = ["AAPL", "MSFT", "GOOGL"]
            else:
                STOCKS = json.loads(content)
    except (json.JSONDecodeError, IOError):
        st.error(f"⚠️ stocks.json error. Using defaults.")
        STOCKS = ["AAPL", "MSFT", "GOOGL"]
else:
    STOCKS = ["AAPL", "MSFT", "GOOGL"]

tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Overview", "🔍 Deep Dive", "🤖 AI Recommendations", "📈 Observability"])

# ====================== TAB 1: OVERVIEW ======================
with tab1:
    col_title, col_button = st.columns([4, 1])
    with col_title:
        st.subheader("All Stocks Overview")
        st.caption(f"⏱️ **Data last retrieved:** {st.session_state.last_refresh_time}")
        
    with col_button:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_refresh_time = datetime.now().strftime("%d %B %Y, %I:%M:%S %p")
            st.rerun()
    
    # Fetch Data with Company Names and Sparkline data
    df_overview = get_overview_df(STOCKS)
    
    # Display interactive table with Sparklines
    st.dataframe(
        df_overview,
        column_config={
            "Symbol": st.column_config.TextColumn("Ticker", help="Stock shortform"),
            "Company": st.column_config.TextColumn("Company Name", width="large"),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "Change %": st.column_config.NumberColumn("Change %", format="%.2f%%"),
            "30-Day Trend": st.column_config.LineChartColumn("30-Day Trend", y_min=0),
            "Market Cap": "Market Cap",
            "Volume": st.column_config.NumberColumn("Volume", format="%d"),
        },
        use_container_width=True,
        hide_index=True
    )

# ====================== TAB 2: DEEP DIVE ======================
with tab2:
    st.subheader("🔍 Deep Dive Analysis")
    selected_stock = st.selectbox("Select Stock", STOCKS)
    
    info, hist, news = get_stock_data(selected_stock)
    perf = compute_performance(hist)

    col_l, col_r = st.columns([3, 1])
    with col_l:
        if not hist.empty:
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"])])
            fig.update_layout(height=500, template="plotly_white", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
    
    with col_r:
        st.metric("Price", f"${perf.get('current_price')}", f"{perf.get('day_change_pct')}%")
        st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B")
        st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
        st.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")

    st.divider()
    st.subheader("💬 Chat with Assistant")
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about " + selected_stock):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        reply, _ = chat_with_cerebras(prompt, st.session_state.chat_history, st.session_state.api_key, selected_stock)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        save_persistent_data()
        st.rerun()

# ====================== TAB 3: RECOMMENDATIONS ======================
with tab3:
    st.subheader("🤖 Tactical AI Analysis")
    st.info("The AI provides tight entry ranges (±5-10%) based on current prices, not just yearly extremes.")
    
    if st.button("🚀 Run Global Tactical Analysis", type="primary", use_container_width=True):
        all_results = []
        batch_size = 5
        
        # UI Elements for Verbose Loading Messages
        progress_text = st.empty() 
        progress_bar = st.progress(0)
        
        for i in range(0, len(STOCKS), batch_size):
            batch = STOCKS[i : i + batch_size]
            
            # Verbose Message: Showing current progress
            progress_text.markdown(f"**⏳ Analyzing Batch ({i+1} to {min(i+batch_size, len(STOCKS))} of {len(STOCKS)}):** {', '.join(batch)}...")
            
            results = get_batch_recommendations(batch, st.session_state.api_key)
            all_results.extend(results)
            
            progress_bar.progress(min((i + batch_size) / len(STOCKS), 1.0))
            time.sleep(1) # Safety delay
            
        progress_text.success("✅ Analysis Complete!")
        time.sleep(1) 
        progress_text.empty() 
        progress_bar.empty() 
        
        st.session_state.recommendations_df = pd.DataFrame(all_results)
        save_persistent_data()
        st.rerun()

    if "recommendations_df" in st.session_state and not st.session_state.recommendations_df.empty:
        st.dataframe(st.session_state.recommendations_df.astype(str), use_container_width=True)

# ====================== TAB 4: OBSERVABILITY ======================
with tab4:
    st.subheader("📈 LLM Observability")
    if "llm_logs" in st.session_state and st.session_state.llm_logs:
        logs_df = pd.DataFrame(st.session_state.llm_logs)
        st.dataframe(logs_df.astype(str), use_container_width=True)
        st.metric("Total Session Cost", f"${logs_df['cost_usd'].astype(float).sum():.4f}")