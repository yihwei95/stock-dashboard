"""
utils.py (ADVANCED)

Features:
- Market data
- AI (async-ready)
- Signal scoring
- Alert logic
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI
import json
import asyncio
from typing import Dict, Any, List


# ====================== DATA ======================

@st.cache_data(ttl=60)
def get_stock_data(symbol: str):
    """Fetch stock data."""
    ticker = yf.Ticker(symbol)
    return ticker.info, ticker.history(period="1y"), ticker.news[:5]


def compute_performance(hist):
    """Compute price + change."""
    if hist.empty or len(hist) < 2:
        return {"current_price": 0, "day_change_pct": 0}

    current = hist["Close"].iloc[-1]
    prev = hist["Close"].iloc[-2]

    return {
        "current_price": round(current, 2),
        "day_change_pct": round((current - prev) / prev * 100, 2)
    }


def get_overview_df(stocks):
    """Portfolio table."""
    rows = []

    for s in stocks:
        try:
            info, hist, _ = get_stock_data(s)
            perf = compute_performance(hist)

            trend = [round(x, 2) for x in hist["Close"].tail(30)]

            rows.append({
                "Symbol": s,
                "Company": info.get("longName", "N/A"),
                "Price": perf["current_price"],
                "Change %": perf["day_change_pct"],
                "30-Day Trend": trend
            })
        except:
            pass

    return pd.DataFrame(rows)


# ====================== AI ======================

def score_signal(rec: str) -> int:
    """Convert recommendation to score."""
    return {
        "Buy": 90,
        "Neutral": 50,
        "Not Buy": 20
    }.get(rec, 0)


async def async_ai_call(symbol, api_key):
    """Async LLM call per stock."""
    
    client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")

    info, hist, news = get_stock_data(symbol)
    perf = compute_performance(hist)

    payload = {
        "symbol": symbol,
        "price": perf["current_price"],
        "news": [n.get("title") for n in news[:2]]
    }

    prompt = f"""
Return JSON:
{{
"Symbol":"{symbol}",
"Recommendation":"Buy|Neutral|Not Buy",
"BuyRange":"X-Y",
"Reason":"short"
}}

Data:
{json.dumps(payload)}
"""

    try:
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        content = response.choices[0].message.content.strip()

        if "```" in content:
            content = content.split("```")[1]

        parsed = json.loads(content)

        parsed["Score"] = score_signal(parsed.get("Recommendation"))

        return parsed

    except Exception as e:
        return {"Symbol": symbol, "Error": str(e), "Score": 0}


async def run_parallel_ai(stocks, api_key):
    """Run async AI for all stocks."""
    
    tasks = [async_ai_call(s, api_key) for s in stocks]

    return await asyncio.gather(*tasks)


def check_alert(price, buy_range):
    """Check if price near buy zone."""
    try:
        low, high = [float(x) for x in buy_range.split("-")]
        return low * 0.98 <= price <= high * 1.02
    except:
        return False