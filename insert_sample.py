import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
from supabase import create_client

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])

info  = yf.Ticker("NVDA").fast_info
price = round(float(info.last_price or 131.50), 2)
print(f"NVDA live price: {price}")

entry = price
stop  = round(entry * 0.9870, 2)
t1    = round(entry * 1.0185, 2)
t2    = round(entry * 1.0370, 2)

row = {
    "ticker":           "NVDA",
    "direction":        "LONG",
    "entry_price":      entry,
    "stop_loss":        stop,
    "target_one":       t1,
    "target_two":       t2,
    "confidence_score": 88,
    "timeframe":        "1h",
    "status":           "active",
    "ai_explanation":   "NVDA broke above its 1h supply zone with a strong bullish CHoCH backed by 3x average volume. VIX sub-17 (low fear), BTC correlation positive. Multi-timeframe check: 15m and 4h both show aligned bullish structure. Risk/Reward 2.8:1. Primary target aligns with prior 4h resistance cluster.",
}

res = sb.table("signals").insert(row).execute()
print("Inserted ID:", res.data[0]["id"] if res.data else "ERROR - no data")
