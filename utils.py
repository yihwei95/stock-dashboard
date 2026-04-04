"""
utils.py - Advanced Stock Dashboard Utilities

Features:
- Market data fetching (yfinance)
- Performance calculations
- Async AI (Cerebras LLM) calls
- Signal scoring + alerts
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI
import json
import asyncio
from typing import Dict, Any, List

# ====================== DATA FETCHING ======================

@st.cache_data(ttl=60)
def get_stock_data(symbol: str):
    """
    Fetch stock information, historical data (1 year), and news (latest 5 items).
    Cached for 60 seconds to reduce API calls.
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info
    hist = ticker.history(period="1y")
    news = ticker.news[:5]
    return info, hist, news


def compute_performance(hist: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate current price and day change percentage.
    Returns zeros if insufficient data.
    """
    if hist.empty or len(hist) < 2:
        return {"current_price": 0.0, "day_change_pct": 0.0}

    current = hist["Close"].iloc[-1]
    prev = hist["Close"].iloc[-2]
    day_change_pct = ((current - prev) / prev) * 100

    return {"current_price": round(current, 2), "day_change_pct": round(day_change_pct, 2)}


def get_overview_df(stocks: List[str]) -> pd.DataFrame:
    """
    Build main portfolio dataframe with:
    - Symbol, Company Name
    - Current Price, Change %
    - 30-day trend (last 30 closing prices)
    """
    rows = []

    for symbol in stocks:
        try:
            info, hist, _ = get_stock_data(symbol)
            perf = compute_performance(hist)
            trend = [round(x, 2) for x in hist["Close"].tail(30)] if not hist.empty else []

            rows.append({
                "Symbol": symbol,
                "Company": info.get("longName", "N/A"),
                "Price": perf["current_price"],
                "Change %": perf["day_change_pct"],
                "30-Day Trend": trend
            })
        except Exception as e:
            # On failure, add empty row
            rows.append({
                "Symbol": symbol,
                "Company": "N/A",
                "Price": 0.0,
                "Change %": 0.0,
                "30-Day Trend": []
            })

    return pd.DataFrame(rows)

# ====================== AI / SIGNAL SCORING ======================

def score_signal(recommendation: str) -> int:
    """
    Assign numeric score to recommendation:
    - Buy = 90
    - Neutral = 50
    - Not Buy = 20
    """
    return {"Buy": 90, "Neutral": 50, "Not Buy": 20}.get(recommendation, 0)


async def async_ai_call(symbol: str, api_key: str) -> Dict:
    """
    Async call to Cerebras LLM for tactical recommendation.
    Returns dictionary with Symbol, Recommendation, BuyRange, Reason, Score.
    """
    client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")

    info, hist, news = get_stock_data(symbol)
    perf = compute_performance(hist)
    payload = {
        "symbol": symbol,
        "current_price": perf["current_price"],
        "news_titles": [n.get("title") for n in news[:2]]
    }

    # Prompt instructing LLM to return structured JSON
    prompt = f"""
Return JSON ONLY (no markdown):
{{
  "Symbol": "{symbol}",
  "Recommendation": "Buy|Neutral|Not Buy",
  "BuyRange": "X-Y",
  "Reason": "Short rationale"
}}
Data: {json.dumps(payload)}
"""

    try:
        # LLM request
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()

        # Remove code blocks if any
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:].strip()
            content = content.strip()

        parsed = json.loads(content)

        # Add numeric signal score
        parsed["Score"] = score_signal(parsed.get("Recommendation"))

        return parsed

    except Exception as e:
        return {"Symbol": symbol, "Recommendation": "Error", "BuyRange": "0-0", "Reason": str(e), "Score": 0}


async def run_parallel_ai(stocks: List[str], api_key: str) -> List[Dict]:
    """
    Run AI analysis concurrently for all stocks using asyncio.as_completed.
    Returns list of dicts with AI recommendations.
    """
    tasks = [async_ai_call(symbol, api_key) for symbol in stocks]
    results = []
    for task in asyncio.as_completed(tasks):
        res = await task
        results.append(res)
    return results


def check_alert(price: float, buy_range: str) -> bool:
    """
    Check if current price is within ±2% of buy range.
    """
    try:
        low, high = map(float, buy_range.split("-"))
        return low * 0.98 <= price <= high * 1.02
    except:
        return False