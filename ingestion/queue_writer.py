import json
import os
import redis
from dotenv import load_dotenv

load_dotenv()


def get_redis_client():
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True
    )


QUEUE_NAME = os.getenv("REDIS_QUEUE_NAME", "gti:articles")


def push_article(client: redis.Redis, article: dict) -> None:
    """
    Push a single article dict to the Redis queue as JSON.
    LPUSH so the NLP worker can BRPOP from the right (FIFO).
    """
    client.lpush(QUEUE_NAME, json.dumps(article))


def queue_length(client: redis.Redis) -> int:
    return client.llen(QUEUE_NAME)