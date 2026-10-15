import hashlib


def make_article_id(url: str, title: str) -> str:
    """
    SHA-256 hash of URL + title.
    Used as the Redis key to detect duplicates.
    """
    raw = f"{url.strip()}{title.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class InMemoryDedup:
    """
    Simple in-memory seen-set for the current process run.
    Redis handles cross-restart dedup — this handles within-run speed.
    """
    def __init__(self, max_size: int = 10_000):
        self._seen: set = set()
        self._max_size = max_size

    def is_duplicate(self, article_id: str) -> bool:
        return article_id in self._seen

    def mark_seen(self, article_id: str):
        # Evict oldest entries if we hit the size limit
        if len(self._seen) >= self._max_size:
            self._seen = set(list(self._seen)[self._max_size // 2:])
        self._seen.add(article_id)