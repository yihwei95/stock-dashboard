"""
utils.py

Core utility module for the stock dashboard.

Handles:
- Market data retrieval (yfinance)
- Performance calculations
- Portfolio aggregation
- LLM interaction (Cerebras API)
- Observability (logging + cost tracking)
"""

# External libraries for data retrieval and processing
import yfinance as yf
import pandas as pd

# Streamlit (used for caching + session state logging)
import streamlit as st

# Standard utilities
from datetime import datetime
import json
import logging

# OpenAI-compatible client (used with Cerebras endpoint)
from openai import OpenAI

# Type hints for better readability and maintainability
from typing import Dict, Any, List, Tuple


# ====================== LOGGING SETUP ======================

# Configure logging to write into a file for debugging and audit
logging.basicConfig(
    filename='dashboard.log',            # Log file name
    level=logging.INFO,                 # Log level (INFO and above)
    format='%(asctime)s | %(levelname)s | %(message)s'  # Log format
)


# ====================== DATA FETCHING ======================

@st.cache_data(ttl=60)  # Cache results for 60 seconds to reduce API calls
def get_stock_data(symbol: str):
    """
    Fetch stock metadata, historical prices, and recent news.
    """
    ticker = yf.Ticker(symbol)           # Create yfinance ticker object
    
    info = ticker.info                   # Company metadata (dict)
    hist = ticker.history(period="1y")   # 1-year historical OHLC data
    
    news = ticker.news[:8]               # Top 8 news articles
    
    return info, hist, news              # Return all three datasets


def compute_performance(hist: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute latest price and daily percentage change.
    """

    # Handle edge case: insufficient data
    if hist.empty or len(hist) < 2:
        return {"current_price": 0.0, "day_change_pct": 0.0}

    # Get latest closing price
    current = hist["Close"].iloc[-1]

    # Get previous day's closing price
    prev = hist["Close"].iloc[-2]

    # Calculate percentage change
    day_change = ((current - prev) / prev) * 100

    # Return rounded results
    return {
        "current_price": round(current, 2),
        "day_change_pct": round(day_change, 2),
    }


def get_overview_df(stocks: List[str]) -> pd.DataFrame:
    """
    Build portfolio-level dataframe for UI display.
    """

    rows = []  # List to accumulate per-stock data

    for symbol in stocks:
        try:
            # Fetch stock data
            info, hist, _ = get_stock_data(symbol)

            # Compute performance metrics
            perf = compute_performance(hist)

            # Extract company name (fallback if missing)
            full_name = info.get("longName", "Unknown Company")

            # Prefer computed price, fallback to API value
            price = perf["current_price"] or info.get("currentPrice") or 0.0

            # Prefer computed change %, fallback to API value
            change = perf["day_change_pct"] or info.get("regularMarketChangePercent", 0.0)

            # Generate last 30 days trend for sparkline
            trend_data = hist["Close"].tail(30).tolist() if not hist.empty else []

            # Append structured row
            rows.append({
                "Symbol": symbol,
                "Company": full_name,
                "Price": float(price),
                "Change %": float(change),
                "30-Day Trend": trend_data,
                "Volume": info.get("regularMarketVolume", 0),
                "Market Cap": (
                    f"${info.get('marketCap', 0)/1e9:.1f}B"
                    if info.get("marketCap") else "N/A"
                ),
                "PE Ratio": info.get("trailingPE"),
            })

        except Exception:
            # Fail-safe: ensure one bad ticker doesn't break entire dashboard
            rows.append({
                "Symbol": symbol,
                "Company": "N/A",
                "Price": 0.0,
                "Change %": 0.0,
                "30-Day Trend": [],
                "Volume": 0,
                "Market Cap": "N/A",
                "PE Ratio": None
            })

    # Convert list of dicts into DataFrame
    return pd.DataFrame(rows)


# ====================== LLM OBSERVABILITY ======================

def calculate_cost(usage: Dict) -> float:
    """
    Estimate cost based on token usage.
    """

    if not usage:
        return 0.0

    # Sum prompt and completion tokens
    tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

    # Convert to cost (assume $0.10 per 1M tokens)
    return round(tokens / 1_000_000 * 0.10, 4)


def _log_llm_call(target: str, call_type: str, prompt_preview: str, response_preview: str, usage: Dict = None):
    """
    Store LLM call metadata in Streamlit session state.
    """

    # Initialize storage if not present
    if "llm_logs" not in st.session_state:
        st.session_state.llm_logs = []

    # Estimate cost
    cost = calculate_cost(usage)

    # Build structured log entry
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": call_type,
        "target": target,
        "prompt_preview": str(prompt_preview)[:120],
        "response_preview": str(response_preview)[:120],
        "total_tokens": (
            (usage.get("prompt_tokens", 0) if usage else 0) +
            (usage.get("completion_tokens", 0) if usage else 0)
        ),
        "cost_usd": cost,
    }

    # Append to session state
    st.session_state.llm_logs.append(log_entry)


# ====================== AI CORE FUNCTIONS ======================

def get_batch_recommendations(stock_list: List[str], api_key: str) -> List[Dict]:
    """
    Generate tactical buy recommendations using LLM.
    """

    # Exit early if API key missing
    if not api_key:
        return []

    # Initialize client with Cerebras endpoint
    client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")

    batch_data = []

    # Prepare structured input data
    for symbol in stock_list:
        try:
            info, hist, news = get_stock_data(symbol)
            perf = compute_performance(hist)

            batch_data.append({
                "symbol": symbol,
                "current_price": perf.get("current_price"),
                "high_52w": info.get('fiftyTwoWeekHigh'),
                "low_52w": info.get('fiftyTwoWeekLow'),
                "news_titles": [n.get('title') for n in news[:2]]
            })
        except:
            continue  # Skip failures silently

    # Construct prompt for strict JSON output
    prompt = f"""Analyze these stocks and return a JSON array of objects..."""

    try:
        # Send request to LLM
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[
                {"role": "system", "content": "You are a precise financial JSON server. Raw JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Low randomness for consistency
        )

        # Extract content
        content = response.choices[0].message.content.strip()

        # Clean markdown formatting if present
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:].strip()

        # Parse JSON
        parsed = json.loads(content)

        # Extract usage metadata
        usage = response.usage.model_dump() if hasattr(response, "usage") else {}

        # Log call
        _log_llm_call(f"Batch {len(stock_list)}", "Batch Rec", prompt[:100], content[:100], usage)

        return parsed

    except Exception as e:
        # Log failure
        _log_llm_call("Batch", "Error", "Batch request", str(e), {})
        return []


def chat_with_cerebras(user_message: str, history: List[Dict], api_key: str, selected_stock: str) -> Tuple[str, Dict]:
    """
    Chat interface with LLM for a specific stock.
    """

    if not api_key:
        return "API Key Missing", {}

    # Initialize client
    client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")

    # Build conversation context
    messages = (
        [{"role": "system", "content": f"Expert analyst for {selected_stock}"}] +
        history +
        [{"role": "user", "content": user_message}]
    )

    try:
        # Send request
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=messages,
            temperature=0.7  # Higher creativity for chat
        )

        reply = response.choices[0].message.content

        usage = response.usage.model_dump() if hasattr(response, "usage") else {}

        # Log interaction
        _log_llm_call(selected_stock, "Chat", user_message[:100], reply[:100], usage)

        return reply, usage

    except Exception as e:
        return f"Error: {e}", {}