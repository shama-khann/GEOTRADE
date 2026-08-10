import json
import logging
import os
import time
from dotenv import load_dotenv

load_dotenv()

from ingestion.queue_writer import get_redis_client
from nlp.classifier import classify_event
from nlp.sentiment import analyze_sentiment
from nlp.clustering import cluster_articles
from nlp.gti_aggregator import compute_gti
from models.feature_builder import build_feature_vector
from models.lgbm_predictor import predict, load_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

QUEUE_NAME = os.getenv("REDIS_QUEUE_NAME", "gti:articles")
BATCH_SIZE = 20


def process_batch(articles):
    enriched = []
    for article in articles:
        clf  = classify_event(article["title"], article["summary"])
        sent = analyze_sentiment(article["title"], article["summary"])
        enriched.append({**article, **clf, **sent})

    enriched    = cluster_articles(enriched)
    gti_result  = compute_gti(enriched)
    features    = build_feature_vector(gti_result, enriched)
    predictions = predict(features)

    return gti_result, predictions, enriched


def print_report(gti_result, predictions):
    g = gti_result
    p = predictions
    print(f"\n{'='*54}")
    print(f"  GTI SCORE   : {g['gti_score']} / 100")
    print(f"  Volatility  : {'SPIKE EXPECTED' if p['volatility_spike_expected'] else 'Normal'}"
          f"  ({round(p['volatility_spike_probability']*100)}% probability)")
    print(f"  Bias        : {p['directional_bias']}")
    print(f"  Risk level  : {p['risk_level']}")
    print(f"  Stress      : {g['breakdown']['avg_stress_score']}")
    print(f"  Severity    : {g['breakdown']['avg_event_severity']}")
    print(f"\n  Top narratives:")
    for n in g["top_narratives"]:
        print(f"    [{n['article_count']}] {n['label']}")
    print(f"{'='*54}\n")


def main():
    print("Starting GTI full pipeline (NLP + LightGBM)...")
    redis_client = get_redis_client()

    try:
        redis_client.ping()
        print("Redis connected.")
    except Exception as e:
        print(f"Redis not reachable: {e}")
        return

    print("Pre-loading ML models...")
    load_models()
    print("All models ready. Waiting for articles...\n")

    batch = []

    while True:
        item = redis_client.brpop(QUEUE_NAME, timeout=5)

        if item:
            _, raw = item
            article = json.loads(raw)
            batch.append(article)
            log.info(f"[{len(batch)}/{BATCH_SIZE}] {article['title'][:55]}")

        if len(batch) >= BATCH_SIZE or (len(batch) > 0 and item is None):
            log.info(f"Processing batch of {len(batch)}...")
            gti_result, predictions, _ = process_batch(batch)
            print_report(gti_result, predictions)
            batch = []


if __name__ == "__main__":
    main()