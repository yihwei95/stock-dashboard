# utils.py - Enhanced Stock Dashboard Utilities
# Features:
# - Market data fetching
# - Performance calculations
# - Async AI calls with structured safe output
# - Signal scoring + buy price fallback

import yfinance as yf
import pandas as pd
import asyncio
import json
from typing import Dict, Any, List
import numpy as np
from openai import OpenAI

# ====================== DATA FETCHING ======================

def get_stock_data(symbol: str):
    """
    Fetch stock info, historical 1y data, latest 5 news.
    Cached externally in Streamlit app to reduce API calls.
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info
    hist = ticker.history(period="1y")
    news = ticker.news[:5] if hasattr(ticker, "news") else []
    return info, hist, news

def compute_performance(hist: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute latest closing price and day change %
    Returns 0 for empty data to prevent Arrow conversion issues.
    """
    if hist.empty or len(hist) < 2:
        return {"current_price": 0.0, "day_change_pct": 0.0}
    current = hist["Close"].iloc[-1]
    prev = hist["Close"].iloc[-2]
    day_change_pct = ((current - prev)/prev)*100
    return {"current_price": round(current, 2), "day_change_pct": round(day_change_pct, 2)}

# ====================== SIGNAL SCORING ======================

def score_signal(recommendation: str) -> int:
    """
    Map recommendation to numeric score.
    Ensures numeric for Arrow serialization.
    """
    return {"Buy": 90, "Neutral": 50, "Not Buy": 20}.get(recommendation, 0)

# ====================== ASYNC AI CALL ======================

async def async_ai_call(symbol: str, api_key: str) -> Dict[str, Any]:
    """
    Async call to LLM for tactical recommendation.
    Returns Arrow-safe structured dict with:
    Symbol, Recommendation, BuyRange, Reason, Score, PE, MarketCap, DividendYield
    """
    # Create client
    client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")

    # Fetch stock data
    info, hist, news = get_stock_data(symbol)
    perf = compute_performance(hist)

    # Build payload for AI reasoning
    payload = {
        "symbol": symbol,
        "current_price": perf["current_price"],
        "news_titles": [n.get("title") for n in news[:2]] if news else ["No news available"]
    }

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
        # Send async request to LLM
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        # Parse response content
        content = response.choices[0].message.content.strip()

        # Remove code block formatting
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:].strip()
            content = content.strip()

        parsed = json.loads(content)

        # Ensure numeric score
        parsed["Score"] = score_signal(parsed.get("Recommendation", "Neutral"))

        # Ensure buy range string
        if "BuyRange" not in parsed or not parsed["BuyRange"]:
            price = perf["current_price"]
            low = round(price*0.95, 2)
            high = round(price*1.05, 2)
            parsed["BuyRange"] = f"{low}-{high}"

        # Arrow-safe numeric placeholders
        parsed["PE Ratio"] = float(info.get("trailingPE", np.nan) or np.nan)
        parsed["Market Cap"] = float(info.get("marketCap", np.nan) or np.nan)
        parsed["Dividend Yield"] = float(info.get("dividendYield", np.nan) or np.nan)

        # Fallbacks
        parsed["Reason"] = parsed.get("Reason", "No news available")
        parsed["Recommendation"] = parsed.get("Recommendation", "Neutral")

        parsed["Symbol"] = symbol  # ensure symbol consistency

        return parsed

    except Exception as e:
        # Always return structured dict, Arrow-safe
        return {
            "Symbol": symbol,
            "Recommendation": "Error",
            "BuyRange": f"{perf['current_price']}-{perf['current_price']}" if perf['current_price'] else "0-0",
            "Reason": str(e),
            "Score": 0,
            "PE Ratio": np.nan,
            "Market Cap": np.nan,
            "Dividend Yield": np.nan
        }

async def run_parallel_ai(stocks: List[str], api_key: str) -> List[Dict[str, Any]]:
    """
    Run AI analysis concurrently for all stocks using asyncio.as_completed.
    Returns list of structured dicts Arrow-safe.
    """
    tasks = [async_ai_call(s, api_key) for s in stocks]
    results = []
    for task in asyncio.as_completed(tasks):
        res = await task
        results.append(res)
    return results