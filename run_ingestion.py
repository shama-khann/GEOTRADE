import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()

from ingestion.fetcher import fetch_all_sources
from ingestion.queue_writer import get_redis_client, queue_length

log = logging.getLogger(__name__)
INTERVAL = int(os.getenv("FETCH_INTERVAL_SECONDS", 120))


def main():
    print("Starting GTI ingestion service...")
    print(f"Fetch interval: {INTERVAL}s | Queue: gti:articles\n")

    redis_client = get_redis_client()

    # Test Redis connection
    try:
        redis_client.ping()
        print("Redis connected.\n")
    except Exception as e:
        print(f"Redis not reachable: {e}")
        print("Run Redis first (see instructions below).")
        return

    while True:
        fetch_all_sources(redis_client)
        q_len = queue_length(redis_client)
        print(f"Queue depth: {q_len} articles waiting for NLP\n")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()