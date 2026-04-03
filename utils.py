"""
utils.py
Header: Utility module containing core functions for data retrieval, performance calculation, 
LLM observability logging, and API interactions with the Cerebras LLM.
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI
import json
import logging
from typing import Dict, Any, List, Tuple

# ====================== LOGGING SETUP ======================
logging.basicConfig(
    filename='dashboard.log', 
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# ====================== DATA FETCHING ======================
@st.cache_data(ttl=60)
def get_stock_data(symbol: str):
    """Fetches ticker info, 1-year history, and recent news."""
    ticker = yf.Ticker(symbol)
    info = ticker.info
    hist = ticker.history(period="1y")
    news = ticker.news[:8]
    return info, hist, news

def compute_performance(hist: pd.DataFrame) -> Dict[str, Any]:
    """Calculates price and percentage change based on historical data."""
    if hist.empty or len(hist) < 2:
        return {"current_price": 0.0, "day_change_pct": 0.0}
    
    current = hist["Close"].iloc[-1]
    prev = hist["Close"].iloc[-2]
    day_change = ((current - prev) / prev) * 100
    
    perf = {
        "current_price": round(current, 2),
        "day_change_pct": round(day_change, 2),
    }
    return perf

def get_overview_df(stocks: List[str]) -> pd.DataFrame:
    """Builds the main portfolio dataframe including company names and trend data."""
    rows = []
    for symbol in stocks:
        try:
            info, hist, _ = get_stock_data(symbol)
            perf = compute_performance(hist)
            
            # Extract names and prices
            full_name = info.get("longName", "Unknown Company")
            price = perf["current_price"] or info.get("currentPrice") or 0.0
            change = perf["day_change_pct"] or info.get("regularMarketChangePercent", 0.0)
            
            # Trend data for sparkline (last 30 days of closing prices)
            trend_data = hist["Close"].tail(30).tolist() if not hist.empty else []

            rows.append({
                "Symbol": symbol,
                "Company": full_name,
                "Price": float(price),
                "Change %": float(change),
                "30-Day Trend": trend_data,
                "Volume": info.get("regularMarketVolume", 0),
                "Market Cap": f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get("marketCap") else "N/A",
                "PE Ratio": info.get("trailingPE"), 
            })
        except Exception:
            rows.append({
                "Symbol": symbol, "Company": "N/A", "Price": 0.0, "Change %": 0.0, 
                "30-Day Trend": [], "Volume": 0, "Market Cap": "N/A", "PE Ratio": None
            })
    return pd.DataFrame(rows)

# ====================== LLM OBSERVABILITY ======================
def calculate_cost(usage: Dict) -> float:
    if not usage: return 0.0
    tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    return round(tokens / 1_000_000 * 0.10, 4)

def _log_llm_call(target: str, call_type: str, prompt_preview: str, response_preview: str, usage: Dict = None):
    if "llm_logs" not in st.session_state:
        st.session_state.llm_logs = []
    
    cost = calculate_cost(usage)
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": call_type,
        "target": target,
        "prompt_preview": str(prompt_preview)[:120],
        "response_preview": str(response_preview)[:120],
        "total_tokens": (usage.get("prompt_tokens", 0) if usage else 0) + (usage.get("completion_tokens", 0) if usage else 0),
        "cost_usd": cost,
    }
    st.session_state.llm_logs.append(log_entry)

# ====================== AI CORE FUNCTIONS ======================
def get_batch_recommendations(stock_list: List[str], api_key: str) -> List[Dict]:
    if not api_key: return []
    client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")
    
    batch_data = []
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
            continue

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
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[{"role": "system", "content": "You are a precise financial JSON server. Raw JSON only. No markdown."},
                      {"role": "user", "content": prompt}],
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:].strip()
            content = content.strip()
        
        parsed = json.loads(content)
        usage = response.usage.model_dump() if hasattr(response, "usage") else {}
        _log_llm_call(f"Batch {len(stock_list)}", "Batch Rec", prompt[:100], content[:100], usage)
        return parsed
    except Exception as e:
        _log_llm_call("Batch", "Error", "Batch request", str(e), {})
        return []

def chat_with_cerebras(user_message: str, history: List[Dict], api_key: str, selected_stock: str) -> Tuple[str, Dict]:
    if not api_key: return "API Key Missing", {}
    client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")
    messages = [{"role": "system", "content": f"Expert analyst for {selected_stock}"}] + history + [{"role": "user", "content": user_message}]
    try:
        response = client.chat.completions.create(model="llama3.1-8b", messages=messages, temperature=0.7)
        reply = response.choices[0].message.content
        usage = response.usage.model_dump() if hasattr(response, "usage") else {}
        _log_llm_call(selected_stock, "Chat", user_message[:100], reply[:100], usage)
        return reply, usage
    except Exception as e:
        return f"Error: {e}", {}