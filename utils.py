"""
utils.py
Header: Utility module containing core functions for data retrieval, performance calculation, 
LLM observability logging, and API interactions with the Cerebras LLM.
"""

import yfinance as yf # For fetching market data from Yahoo Finance
import pandas as pd # For data manipulation and DataFrame structuring
import streamlit as st # For Streamlit caching and session state management
from datetime import datetime # For timestamping logs
from openai import OpenAI # For interacting with the Cerebras API (OpenAI compatible)
import json # For parsing LLM outputs and handling data structures
import logging # For backend system logging
from typing import Dict, Any, List, Tuple # For type hinting

# ====================== LOGGING SETUP ======================
# Configure the root logger to write to a local file with a specific format
logging.basicConfig(
    filename='dashboard.log', # Log file destination
    level=logging.INFO, # Minimum severity level to log
    format='%(asctime)s | %(levelname)s | %(message)s' # Timestamp, severity, and message format
)

# ====================== DATA FETCHING ======================
# Cache the data for 30 seconds to prevent rate-limiting from Yahoo Finance
@st.cache_data(ttl=30)
def get_stock_data(symbol: str):
    # Initialize the Ticker object for the requested symbol
    ticker = yf.Ticker(symbol)
    # Fetch general company information and current metrics
    info = ticker.info
    # Fetch 1 year of historical price data
    hist = ticker.history(period="1y")
    # Fetch the 8 most recent news articles related to the ticker
    news = ticker.news[:8]
    # Return a tuple containing info, historical data, and news
    return info, hist, news

def compute_performance(hist: pd.DataFrame) -> Dict[str, Any]:
    # Check if the historical dataframe is empty or lacks enough data to compare
    if hist.empty or len(hist) < 2:
        # Return zeroed metrics if data is insufficient
        return {"current_price": 0.0, "day_change_pct": 0.0}
    
    # Extract the most recent closing price
    current = hist["Close"].iloc[-1]
    # Extract the previous day's closing price
    prev = hist["Close"].iloc[-2]
    # Calculate the percentage change between the two days
    day_change = ((current - prev) / prev) * 100
    
    # Format the calculated metrics into a dictionary
    perf = {
        "current_price": round(current, 2), # Round price to 2 decimals
        "day_change_pct": round(day_change, 2), # Round percentage to 2 decimals
    }
    # Return the performance dictionary
    return perf

def get_overview_df(stocks: List[str]) -> pd.DataFrame:
    # Initialize an empty list to store row dictionaries
    rows = []
    # Iterate through each stock symbol in the provided list
    for symbol in stocks:
        try:
            # Fetch data tuple (ignoring news with '_')
            info, hist, _ = get_stock_data(symbol)
            # Compute day-to-day performance
            perf = compute_performance(hist)
            # Determine current price, prioritizing calculated price over info dictionary
            price = perf["current_price"] or info.get("currentPrice") or 0.0
            # Determine percentage change, prioritizing calculated change over info dictionary
            change = perf["day_change_pct"] or info.get("regularMarketChangePercent", 0.0)
            
            # Safely extract the trailing Price-to-Earnings ratio
            pe = info.get("trailingPE")
            # If PE exists, convert to float and round to 1 decimal
            if pe: pe = round(float(pe), 1)

            # Append the processed stock data as a dictionary to the rows list
            rows.append({
                "Symbol": symbol,
                "Price": float(price),
                "Change %": float(change),
                "Volume": info.get("regularMarketVolume", 0),
                # Format Market Cap to Billions with a '$' prefix and 'B' suffix, or N/A if missing
                "Market Cap": f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get("marketCap") else "N/A",
                "PE Ratio": pe, 
            })
        except Exception:
            # If any error occurs (e.g., delisted stock), append default fallback values
            rows.append({"Symbol": symbol, "Price": 0.0, "Change %": 0.0, "Volume": 0, "Market Cap": "N/A", "PE Ratio": None})
    # Convert the list of dictionaries into a Pandas DataFrame and return it
    return pd.DataFrame(rows)

# ====================== LLM OBSERVABILITY ======================
def calculate_cost(usage: Dict) -> float:
    # Return 0 if no usage dictionary is provided
    if not usage: return 0.0
    # Sum the prompt tokens and completion tokens
    tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    # Calculate estimated cost (assuming $0.10 per 1M tokens) and round to 4 decimals
    return round(tokens / 1_000_000 * 0.10, 4)

def _log_llm_call(target: str, call_type: str, prompt_preview: str, response_preview: str, usage: Dict = None):
    # Initialize the llm_logs list in Streamlit session state if it doesn't exist
    if "llm_logs" not in st.session_state:
        st.session_state.llm_logs = []
    
    # Calculate the financial cost of the LLM call
    cost = calculate_cost(usage)
    # Create a dictionary representing the log entry
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Current time
        "type": call_type, # Type of call (e.g., 'Chat', 'Batch Rec')
        "target": target, # The specific stock or batch identifier
        "prompt_preview": str(prompt_preview)[:120], # Truncate prompt to 120 chars
        "response_preview": str(response_preview)[:120], # Truncate response to 120 chars
        # Calculate total tokens safely handling None values
        "total_tokens": (usage.get("prompt_tokens", 0) if usage else 0) + (usage.get("completion_tokens", 0) if usage else 0),
        "cost_usd": cost, # Appended cost
    }
    # Append the new log entry to the session state log array
    st.session_state.llm_logs.append(log_entry)

# ====================== AI CORE FUNCTIONS ======================
def get_batch_recommendations(stock_list: List[str], api_key: str) -> List[Dict]:
    # Return an empty list if no API key is provided
    if not api_key: return []
    # Initialize the OpenAI client pointing to the Cerebras base URL
    client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")
    
    # Initialize a list to hold the prepped data for the LLM
    batch_data = []
    # Loop through the requested symbols to gather context data
    for symbol in stock_list:
        try:
            # Fetch raw data
            info, hist, news = get_stock_data(symbol)
            # Calculate current performance
            perf = compute_performance(hist)
            # Append a simplified context dictionary for the LLM prompt
            batch_data.append({
                "symbol": symbol,
                "current_price": perf.get("current_price"),
                "high_52w": info.get('fiftyTwoWeekHigh'),
                "low_52w": info.get('fiftyTwoWeekLow'),
                # Extract just the titles from the top 2 news articles
                "news_titles": [n.get('title') for n in news[:2]]
            })
        except:
            # Skip this symbol if data fetching fails
            continue

    # Construct the strict prompt instructing the LLM to output raw JSON
    prompt = f"""Analyze these stocks and return a JSON array of objects. 
For each stock, suggest a TACTICAL buy range based on the 'current_price'. 
Do NOT just return the 52-week range. Provide a tight entry range (e.g., +/- 5% of current).

Data: {json.dumps(batch_data)}

Output format (Raw JSON Array):
[{{
  "Symbol": "TICKER",
  "Current Price": 0.00,
  "Tactical Buy Range": "$X - $Y",
  "Recommendation": "Buy/Neutral/Not Buy",
  "Reason": "Short technical reason"
}}]"""

    try:
        # Execute the chat completion call to Llama 3.1 8B via Cerebras
        response = client.chat.completions.create(
            model="llama3.1-8b",
            # System prompt enforces raw JSON behavior
            messages=[{"role": "system", "content": "You are a precise financial JSON server. Raw JSON only. No markdown."},
                      {"role": "user", "content": prompt}],
            temperature=0.1 # Low temperature for deterministic, analytical output
        )
        # Extract the text content from the response
        content = response.choices[0].message.content.strip()
        
        # Clean up potential markdown code blocks (e.g., ```json ... ```)
        if "```" in content:
            # Split by backticks and take the inner content
            content = content.split("```")[1]
            # Remove the 'json' language specifier if present
            if content.startswith("json"):
                content = content[4:].strip()
            # Strip remaining whitespace
            content = content.strip()
        
        # Parse the cleaned string into Python objects
        parsed = json.loads(content)
        # Extract usage statistics if the attribute exists
        usage = response.usage.model_dump() if hasattr(response, "usage") else {}
        # Log the call metrics
        _log_llm_call(f"Batch {len(stock_list)}", "Batch Rec", prompt[:100], content[:100], usage)
        # Return the parsed JSON list
        return parsed
    except Exception as e:
        # If parsing or API call fails, log the error and return an empty list
        _log_llm_call("Batch", "Error", "Batch request", str(e), {})
        return []

def chat_with_cerebras(user_message: str, history: List[Dict], api_key: str, selected_stock: str) -> Tuple[str, Dict]:
    # Return error string and empty usage if API key is missing
    if not api_key: return "API Key Missing", {}
    # Initialize the OpenAI client pointing to the Cerebras base URL
    client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")
    # Construct the message array: System prompt + previous history + current user message
    messages = [{"role": "system", "content": f"Expert analyst for {selected_stock}"}] + history + [{"role": "user", "content": user_message}]
    try:
        # Execute the chat completion call with higher temperature for conversational variance
        response = client.chat.completions.create(model="llama3.1-8b", messages=messages, temperature=0.7)
        # Extract the assistant's reply
        reply = response.choices[0].message.content
        # Extract usage statistics
        usage = response.usage.model_dump() if hasattr(response, "usage") else {}
        # Log the conversational call
        _log_llm_call(selected_stock, "Chat", user_message[:100], reply[:100], usage)
        # Return the reply string and usage dictionary
        return reply, usage
    except Exception as e:
        # Return formatted error if the API call fails
        return f"Error: {e}", {}