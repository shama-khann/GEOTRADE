EVENT_TYPE_MAP = {
    "military conflict":      0,
    "economic sanctions":     1,
    "trade dispute":          2,
    "energy crisis":          3,
    "political instability":  4,
    "central bank policy":    5,
    "natural disaster":       6,
    "diplomatic tension":     7,
}

REGION_MAP = {
    "global":      0,
    "middle_east": 1,
    "europe":      2,
    "asia":        3,
    "africa":      4,
    "americas":    5,
    "oceania":     6,
}

BIAS_LABELS    = {0: "Bullish", 1: "Neutral", 2: "Bearish"}
RISK_LABELS    = {0: "Low", 1: "Elevated", 2: "High", 3: "Severe"}


def build_feature_vector(gti_result: dict, articles: list[dict]) -> dict:
    """
    Takes the GTI aggregator output + enriched articles,
    returns a flat feature dict ready for LightGBM.
    """
    breakdown = gti_result.get("breakdown", {})

    # Most common event type in this batch
    event_types = [a.get("event_type", "diplomatic tension") for a in articles]
    top_event   = max(set(event_types), key=event_types.count)

    # Most common region
    regions     = [a.get("region", "global") for a in articles]
    top_region  = max(set(regions), key=regions.count)

    return {
        "gti_score":       gti_result.get("gti_score", 0) / 100,
        "avg_stress":      breakdown.get("avg_stress_score", 0.5),
        "avg_severity":    breakdown.get("avg_event_severity", 0.5),
        "cluster_density": breakdown.get("cluster_density", 0.5),
        "event_type":      EVENT_TYPE_MAP.get(top_event, 7),
        "region":          REGION_MAP.get(top_region, 0),
    }