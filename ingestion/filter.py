import re

# Articles must contain at least one of these to pass
GEOPOLITICAL_KEYWORDS = [
    # Conflict & military
    "war", "conflict", "missile", "sanctions", "troops", "military",
    "ceasefire", "airstrike", "nuclear", "weapons", "invasion", "coup",
    # Economics & markets
    "tariff", "trade war", "inflation", "interest rate", "federal reserve",
    "recession", "gdp", "oil", "gas", "energy", "commodities", "currency",
    "central bank", "imf", "world bank", "debt",
    # Geopolitics
    "nato", "un security council", "g7", "g20", "opec", "sanctions",
    "china", "russia", "iran", "north korea", "taiwan", "ukraine",
    "middle east", "israel", "gaza", "india", "pakistan",
    # Markets
    "stock market", "volatility", "rally", "selloff", "crash", "hedge",
    "bond yield", "dollar", "euro", "yuan", "supply chain",
]

# Hard drop — skip immediately if any of these appear
NOISE_KEYWORDS = [
    "sports", "football", "soccer", "nba", "nfl", "celebrity",
    "entertainment", "movie", "music", "fashion", "recipe",
    "lifestyle", "travel tips", "horoscope",
]

# Compile once at import time for speed
_GEO_PATTERN = re.compile(
    "|".join(re.escape(k) for k in GEOPOLITICAL_KEYWORDS),
    re.IGNORECASE
)
_NOISE_PATTERN = re.compile(
    "|".join(re.escape(k) for k in NOISE_KEYWORDS),
    re.IGNORECASE
)


def is_relevant(title: str, summary: str) -> bool:
    """
    Returns True if the article should pass to the NLP queue.
    Fast check — no ML involved, just regex.
    """
    text = f"{title} {summary}"

    # Hard drop first (cheapest check)
    if _NOISE_PATTERN.search(text):
        return False

    # Must match at least one geopolitical keyword
    return bool(_GEO_PATTERN.search(text))