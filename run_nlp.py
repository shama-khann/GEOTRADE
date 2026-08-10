import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()

from ingestion.queue_writer import get_redis_client
from nlp.classifier import classify_event
from nlp.sentiment import analyze_sentiment
from nlp.clustering import cluster_articles
from nlp.gti_aggregator import compute_gti

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

QUEUE_NAME = os.getenv("REDIS_QUEUE_NAME", "gti:articles")
BATCH_SIZE  = 20  # Process articles in batches for clustering


def process_batch(articles: list[dict]) -> dict:
    enriched = []
    for article in articles:
        log.info(f"Processing: {article['title'][:60]}...")

        classification = classify_event(article["title"], article["summary"])
        sentiment      = analyze_sentiment(article["title"], article["summary"])

        enriched.append({
            **article,
            **classification,
            **sentiment,
        })

    # Cluster the whole batch together
    enriched = cluster_articles(enriched)

    # Compute GTI for this batch
    gti = compute_gti(enriched)

    return {"articles": enriched, "gti": gti}


def main():
    print("Starting GTI NLP worker...")
    redis_client = get_redis_client()

    try:
        redis_client.ping()
        print("Redis connected.\n")
    except Exception as e:
        print(f"Redis not reachable: {e}")
        return

    print("Loading ML models (first run will download models ~700MB total)...")
    print("Waiting for articles from queue...\n")

    batch = []

    while True:
        # BRPOP blocks until an article is available (timeout 5s)
        item = redis_client.brpop(QUEUE_NAME, timeout=5)

        if item:
            _, raw = item
            article = json.loads(raw)
            batch.append(article)
            log.info(f"Got article [{len(batch)}/{BATCH_SIZE}]: {article['title'][:50]}")

        # Process when batch is full or queue is empty
        if len(batch) >= BATCH_SIZE or (len(batch) > 0 and item is None):
            log.info(f"\nProcessing batch of {len(batch)} articles...")
            result = process_batch(batch)

            gti = result["gti"]
            print(f"\n{'='*50}")
            print(f"GTI SCORE: {gti['gti_score']} / 100")
            print(f"Articles processed: {gti['breakdown']['article_count']}")
            print(f"Avg stress: {gti['breakdown']['avg_stress_score']}")
            print(f"Avg severity: {gti['breakdown']['avg_event_severity']}")
            print(f"\nTop narratives:")
            for n in gti["top_narratives"]:
                print(f"  [{n['article_count']} articles] {n['label']}")
            print(f"{'='*50}\n")

            batch = []


if __name__ == "__main__":
    main()