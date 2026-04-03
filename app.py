"""
app.py
Header: The main Streamlit entrypoint. Handles UI layout, state initialization, 
secrets management, and rendering data visualization components.
"""

import streamlit as st # Core UI library
import json # For reading local config files
import pandas as pd # For structuring table views
import plotly.graph_objects as go # For rendering interactive candlestick charts
import os # For file path and directory operations
import time # For simulating progress bar delays
from datetime import datetime # For potential UI timestamping

# Import custom utility functions from utils.py
from utils import (
    get_stock_data, get_overview_df, get_batch_recommendations,
    chat_with_cerebras, compute_performance
)

# Configure the Streamlit page metadata and set layout to take full screen width
st.set_page_config(page_title="Pro Stock Dashboard", page_icon="📈", layout="wide")

# Ensure the local data directory exists for file operations 
# Note: On Streamlit Community Cloud, this directory is ephemeral and will reset on container restart.
os.makedirs("data", exist_ok=True)

def load_persistent_data():
    # Attempt to load previously saved LLM recommendations from a local JSON file
    if os.path.exists("data/recommendations.json"):
        try:
            # Read JSON into a Pandas DataFrame in session state
            st.session_state.recommendations_df = pd.read_json("data/recommendations.json")
        except:
            # Fallback to empty DataFrame on read failure
            st.session_state.recommendations_df = pd.DataFrame()
            
    # Attempt to load previous chat history from a local JSON file
    if "chat_history" not in st.session_state and os.path.exists("data/chat_history.json"):
        try:
            with open("data/chat_history.json", "r") as f:
                # Parse JSON array into session state list
                st.session_state.chat_history = json.load(f)
        except:
            # Fallback to empty list on read failure
            st.session_state.chat_history = []

def save_persistent_data():
    # Save the current recommendations DataFrame to a local JSON file
    if "recommendations_df" in st.session_state:
        st.session_state.recommendations_df.to_json("data/recommendations.json", orient="records")
        
    # Save the current chat history list to a local JSON file
    if "chat_history" in st.session_state:
        with open("data/chat_history.json", "w") as f:
            json.dump(st.session_state.chat_history, f)

# Execute the loading function on app start
load_persistent_data()

# Render main title
st.title("📈 Professional Stock Dashboard")

# ====================== SIDEBAR ======================
# Setup sidebar header
st.sidebar.title("⚙️ Settings")
# Add a radio button for theme selection
theme_choice = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)

# Initialize theme in session state if missing
if "theme" not in st.session_state: st.session_state.theme = "dark"

# If the user changes the theme, update state and trigger a UI rerun
if theme_choice.lower() != st.session_state.theme:
    st.session_state.theme = theme_choice.lower()
    st.rerun()

# Apply custom CSS if dark theme is selected
if st.session_state.theme == "dark":
    st.markdown("<style>.stApp { background-color: #0e1117; color: #fafafa; }</style>", unsafe_allow_html=True)

# ===== NEW: Streamlit Secrets Integration =====
# Check if API key needs to be initialized
if "api_key" not in st.session_state:
    # Safely attempt to load from Streamlit Secrets (ideal for Cloud deployment)
    if "CEREBRAS_API_KEY" in st.secrets:
        st.session_state.api_key = st.secrets["CEREBRAS_API_KEY"]
    else:
        # Fallback to empty string if no secret exists
        st.session_state.api_key = ""

# Render an input field in the sidebar for the user to view/edit the API key
api_key = st.sidebar.text_input("Cerebras API Key", value=st.session_state.api_key, type="password")
# If the user types a new key, update the session state
if api_key: st.session_state.api_key = api_key

# Load the list of tracked stocks from the local configuration file
if os.path.exists("stocks.json"):
    with open("stocks.json", "r") as f:
        STOCKS = json.load(f)
else:
    # Fallback to default mega-caps if file is missing
    STOCKS = ["AAPL", "MSFT", "GOOGL"]

# Create 4 logical tabs in the main UI area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Overview", "🔍 Deep Dive", "🤖 AI Recommendations", "📈 Observability"])

# ====================== TAB 1: OVERVIEW ======================
with tab1:
    st.subheader("All Stocks Overview")
    # Button to clear the data cache and force a fresh fetch from yfinance
    if st.button("🔄 Refresh Portfolio", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Fetch aggregated data for all stocks
    df_overview = get_overview_df(STOCKS)
    # Render table. Converting to string prevents PyArrow serialization crashes on mixed types.
    st.dataframe(df_overview.astype(str), use_container_width=True)

    # Accordion element for documentation
    with st.expander("📖 Stock Metric Legend"):
        st.markdown("- **Price**: Latest price\n- **Change %**: Daily change\n- **Volume**: Shares traded\n- **Market Cap**: Total value\n- **PE Ratio**: Price-to-Earnings")

# ====================== TAB 2: DEEP DIVE ======================
with tab2:
    st.subheader("🔍 Deep Dive Analysis")
    # Dropdown to select a specific stock for detailed view
    selected_stock = st.selectbox("Select Stock", STOCKS)
    
    # Fetch specific details for the chosen stock
    info, hist, news = get_stock_data(selected_stock)
    perf = compute_performance(hist)

    # Split layout into wide left column (chart) and narrow right column (metrics)
    col_l, col_r = st.columns([3, 1])
    with col_l:
        # Ensure historical data exists before plotting
        if not hist.empty:
            # Build a Plotly Candlestick chart
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"])])
            # Apply layout themes based on the app's current theme state
            fig.update_layout(height=500, template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white")
            # Render chart natively in Streamlit
            st.plotly_chart(fig, use_container_width=True)
    
    with col_r:
        # Display key metrics as large callout numbers
        st.metric("Price", f"${perf.get('current_price')}", f"{perf.get('day_change_pct')}%")
        st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B")

    # Visual separator
    st.divider()
    st.subheader("💬 Chat with Assistant")
    # Initialize local chat history for this session if missing
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    
    # Re-render all previous chat messages in the UI
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    # Wait for the user to submit a new prompt via the chat input
    if prompt := st.chat_input("Ask about " + selected_stock):
        # Append user message to state and render it
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # Trigger the LLM API call
        reply, _ = chat_with_cerebras(prompt, st.session_state.chat_history, st.session_state.api_key, selected_stock)
        
        # Append assistant reply to state and save
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        save_persistent_data()
        # Force rerun to show the assistant's message correctly in sequence
        st.rerun()

# ====================== TAB 3: RECOMMENDATIONS ======================
with tab3:
    st.subheader("🤖 Tactical AI Analysis")
    st.info("The AI provides tight entry ranges (±5-10%) based on current prices, not just yearly extremes.")
    
    # Action button to trigger batch processing
    if st.button("🚀 Run Global Tactical Analysis", type="primary", use_container_width=True):
        all_results = []
        batch_size = 5 # Process stocks in groups of 5 to manage prompt length
        progress_bar = st.progress(0) # Initialize visual progress bar
        
        # Loop through the stocks list in chunks
        for i in range(0, len(STOCKS), batch_size):
            batch = STOCKS[i : i + batch_size]
            # Fetch JSON recommendations from Cerebras LLM
            results = get_batch_recommendations(batch, st.session_state.api_key)
            all_results.extend(results)
            # Update the progress bar mathematically based on completed chunks
            progress_bar.progress(min((i + batch_size) / len(STOCKS), 1.0))
            # Sleep briefly to avoid aggressive rate-limiting
            time.sleep(1)
            
        # Convert the aggregated list of dicts to a DataFrame and store in state
        st.session_state.recommendations_df = pd.DataFrame(all_results)
        save_persistent_data()
        st.rerun()

    # Display the results table if data exists
    if "recommendations_df" in st.session_state and not st.session_state.recommendations_df.empty:
        st.dataframe(st.session_state.recommendations_df.astype(str), use_container_width=True)

# ====================== TAB 4: OBSERVABILITY ======================
with tab4:
    st.subheader("📈 LLM Observability")
    # Check if there are active logs generated during this session
    if "llm_logs" in st.session_state and st.session_state.llm_logs:
        # Convert logs list into DataFrame
        logs_df = pd.DataFrame(st.session_state.llm_logs)
        # Display logs table
        st.dataframe(logs_df.astype(str), use_container_width=True)
        # Calculate and display the total estimated API cost incurred in the current session
        st.metric("Total Session Cost", f"${logs_df['cost_usd'].astype(float).sum():.4f}")