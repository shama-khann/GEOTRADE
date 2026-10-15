from transformers import pipeline
import logging

log = logging.getLogger(__name__)

# Lightweight model — good CPU performance
MODEL = "facebook/bart-large-mnli"

EVENT_LABELS = [
    "military conflict",
    "economic sanctions",
    "trade dispute",
    "energy crisis",
    "political instability",
    "central bank policy",
    "natural disaster",
    "diplomatic tension",
]

_classifier = None


def load_classifier():
    global _classifier
    if _classifier is None:
        log.info("Loading event classifier model (first run downloads ~500MB)...")
        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL,
            device=-1  # CPU
        )
        log.info("Classifier ready.")
    return _classifier


def classify_event(title: str, summary: str) -> dict:
    """
    Returns the top event type and confidence score.
    """
    clf = load_classifier()
    text = f"{title}. {summary[:300]}"  # Cap at 300 chars for speed

    result = clf(text, EVENT_LABELS, multi_label=False)

    return {
        "event_type": result["labels"][0],
        "confidence": round(result["scores"][0], 4),
        "all_scores": dict(zip(result["labels"], [round(s, 4) for s in result["scores"]]))
    }