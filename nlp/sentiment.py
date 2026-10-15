from transformers import pipeline
import logging

log = logging.getLogger(__name__)

# Fine-tuned on financial news — much better than generic sentiment
MODEL = "ProsusAI/finbert"

_sentiment_pipe = None


def load_sentiment():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        log.info("Loading FinBERT sentiment model...")
        _sentiment_pipe = pipeline(
            "text-classification",
            model=MODEL,
            device=-1  # CPU
        )
        log.info("Sentiment model ready.")
    return _sentiment_pipe


def analyze_sentiment(title: str, summary: str) -> dict:
    """
    Returns sentiment label and a normalized stress score.
    Stress score: 1.0 = maximum geopolitical stress, 0.0 = calm
    """
    pipe = load_sentiment()
    text = f"{title}. {summary[:512]}"

    result = pipe(text, truncation=True, max_length=512)[0]
    label = result["label"].lower()   # positive / negative / neutral
    score = round(result["score"], 4)

    # Convert to stress score (negative news = high stress)
    if label == "negative":
        stress = round(score, 4)
    elif label == "positive":
        stress = round(1.0 - score, 4)
    else:
        stress = 0.5

    return {
        "sentiment_label": label,
        "sentiment_score": score,
        "stress_score": stress
    }