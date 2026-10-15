import logging

log = logging.getLogger(__name__)

# Weight of each signal in the final GTI score
WEIGHTS = {
    "stress_score":      0.45,  # Sentiment stress (most important)
    "event_severity":    0.35,  # How severe the event type is
    "cluster_density":   0.20,  # How many articles in this cluster
}

# How severe each event type is (0.0 to 1.0)
EVENT_SEVERITY = {
    "military conflict":      1.0,
    "economic sanctions":     0.85,
    "diplomatic tension":     0.70,
    "energy crisis":          0.75,
    "political instability":  0.65,
    "trade dispute":          0.55,
    "central bank policy":    0.40,
    "natural disaster":       0.50,
}


def compute_gti(processed_articles: list[dict]) -> dict:
    """
    Takes a list of fully processed articles (with classifier +
    sentiment + cluster fields) and returns the GTI score and breakdown.
    """
    if not processed_articles:
        return {"gti_score": 0, "breakdown": {}, "top_narratives": []}

    total_stress   = 0.0
    total_severity = 0.0
    cluster_sizes  = {}

    for article in processed_articles:
        total_stress   += article.get("stress_score", 0.5)
        event_type      = article.get("event_type", "diplomatic tension")
        total_severity += EVENT_SEVERITY.get(event_type, 0.5)

        cid = article.get("cluster_id", 0)
        cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

    n = len(processed_articles)
    avg_stress   = total_stress / n
    avg_severity = total_severity / n

    # Cluster density: how concentrated are narratives (0–1)
    max_cluster  = max(cluster_sizes.values()) if cluster_sizes else 1
    density      = min(1.0, max_cluster / max(n, 1) * 2)

    raw_score = (
        avg_stress   * WEIGHTS["stress_score"] +
        avg_severity * WEIGHTS["event_severity"] +
        density      * WEIGHTS["cluster_density"]
    )

    gti_score = round(min(100, raw_score * 100), 1)

    # Top narratives = largest clusters
    top_narratives = sorted(
        [
            {
                "cluster_id": cid,
                "article_count": count,
                "label": next(
                    (a["cluster_label"] for a in processed_articles
                     if a.get("cluster_id") == cid),
                    "unknown"
                )
            }
            for cid, count in cluster_sizes.items()
        ],
        key=lambda x: x["article_count"],
        reverse=True
    )[:5]

    return {
        "gti_score": gti_score,
        "breakdown": {
            "avg_stress_score":   round(avg_stress, 4),
            "avg_event_severity": round(avg_severity, 4),
            "cluster_density":    round(density, 4),
            "article_count":      n,
        },
        "top_narratives": top_narratives
    }