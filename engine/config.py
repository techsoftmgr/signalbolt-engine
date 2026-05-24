from dotenv import load_dotenv
import os
import sys

load_dotenv(override=True)   # .env values always win over shell environment


def get_required(key: str) -> str:
    value = os.getenv(key)
    if not value:
        print(f"ERROR: Missing required key: {key}")
        print(f"Copy .env.example to .env and add your keys")
        sys.exit(1)
    return value


SUPABASE_URL        = get_required("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_KEY") or get_required("SUPABASE_SECRET_KEY")
ANTHROPIC_API_KEY   = get_required("ANTHROPIC_API_KEY")

ALPACA_API_KEY    = get_required("ALPACA_API_KEY")
ALPACA_SECRET_KEY = get_required("ALPACA_SECRET_KEY")
ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_FEED  = os.getenv("ALPACA_DATA_FEED", "iex")   # "sip" on paid plan, "iex" on free

STRIPE_SECRET_KEY      = get_required("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET  = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRO_PRICE_ID    = os.getenv("STRIPE_PRO_PRICE_ID", "")
STRIPE_PRO_PLUS_PRICE_ID = os.getenv("STRIPE_PRO_PLUS_PRICE_ID", "")

STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
SENTRY_DSN             = os.getenv("SENTRY_DSN", "")
ENVIRONMENT            = os.getenv("ENVIRONMENT", "production")
ENGINE_PUBLIC_URL      = os.getenv("ENGINE_PUBLIC_URL", "http://localhost:8000")
UNUSUAL_WHALES_API_KEY = os.getenv("UNUSUAL_WHALES_API_KEY", "")
POLYGON_API_KEY        = os.getenv("POLYGON_API_KEY", "")
EXPO_ACCESS_TOKEN      = os.getenv("EXPO_ACCESS_TOKEN", "")
PORT                   = int(os.getenv("PORT", "8000"))
# Internal engine API key — protects /run and /inject-test-signal from public access
ENGINE_API_KEY         = os.getenv("ENGINE_API_KEY", "")

# ── Premium feature flags (default on — flip to "false" in .env to disable) ───
ENABLE_HEATMAP         = os.getenv("ENABLE_HEATMAP",         "true").lower() == "true"
ENABLE_QUANT_DASHBOARD = os.getenv("ENABLE_QUANT_DASHBOARD", "true").lower() == "true"
ENABLE_NEWS_REACTION   = os.getenv("ENABLE_NEWS_REACTION",   "true").lower() == "true"
ENABLE_SOCIAL_SIGNALS  = os.getenv("ENABLE_SOCIAL_SIGNALS",  "true").lower() == "true"
