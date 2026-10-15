import feedparser
import logging
import time
from datetime import datetime, timezone

from ingestion.sources import SOURCES
from ingestion.filter import is_relevant
from ingestion.dedup import make_article_id, InMemoryDedup
from ingestion.queue_writer import push_article, queue_length, get_redis_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

dedup = InMemoryDedup()


def fetch_all_sources(redis_client) -> dict:
    """
    Fetch all RSS sources once. Returns a summary dict.
    """
    stats = {"fetched": 0, "passed_filter": 0, "duplicates": 0, "queued": 0}

    for source in SOURCES:
        log.info(f"Fetching: {source['name']}")

        try:
            feed = feedparser.parse(source["url"])
        except Exception as e:
            log.warning(f"Failed to fetch {source['name']}: {e}")
            continue

        for entry in feed.entries:
            stats["fetched"] += 1

            title   = getattr(entry, "title",   "")
            summary = getattr(entry, "summary", "")
            url     = getattr(entry, "link",    "")
            pub     = getattr(entry, "published", datetime.now(timezone.utc).isoformat())

            # 1. Keyword filter (cheap — runs before dedup)
            if not is_relevant(title, summary):
                continue
            stats["passed_filter"] += 1

            # 2. Dedup check
            article_id = make_article_id(url, title)
            if dedup.is_duplicate(article_id):
                stats["duplicates"] += 1
                continue
            dedup.mark_seen(article_id)

            # 3. Build clean article object
            article = {
                "id":        article_id,
                "source":    source["name"],
                "region":    source["region"],
                "title":     title,
                "summary":   summary,
                "url":       url,
                "published": pub,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }

            # 4. Push to Redis queue
            push_article(redis_client, article)
            stats["queued"] += 1

    log.info(
        f"Cycle complete — fetched: {stats['fetched']} | "
        f"passed filter: {stats['passed_filter']} | "
        f"dupes skipped: {stats['duplicates']} | "
        f"queued: {stats['queued']}"
    )
    return stats