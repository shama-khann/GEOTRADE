from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

log = logging.getLogger(__name__)

MODEL = "all-MiniLM-L6-v2"  # Fast, small, excellent quality

_embedder = None


def load_embedder():
    global _embedder
    if _embedder is None:
        log.info("Loading sentence embedder...")
        _embedder = SentenceTransformer(MODEL)
        log.info("Embedder ready.")
    return _embedder


def embed_articles(articles: list[dict]) -> np.ndarray:
    embedder = load_embedder()
    texts = [f"{a['title']}. {a['summary'][:200]}" for a in articles]
    return embedder.encode(texts, show_progress_bar=False)


def cluster_articles(articles: list[dict], n_clusters: int = None) -> list[dict]:
    """
    Clusters a batch of articles into macro narratives.
    Returns articles with cluster_id and cluster_label added.
    """
    if len(articles) < 3:
        for a in articles:
            a["cluster_id"] = 0
            a["cluster_label"] = "unclustered"
        return articles

    embeddings = embed_articles(articles)

    # Auto-pick cluster count if not specified
    if n_clusters is None:
        n_clusters = max(2, min(7, len(articles) // 8))

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    # Find the most central article per cluster to use as label
    cluster_labels = {}
    for cluster_id in range(n_clusters):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) == 0:
            continue
        center = km.cluster_centers_[cluster_id]
        cluster_embeddings = embeddings[indices]
        sims = cosine_similarity([center], cluster_embeddings)[0]
        best_idx = indices[np.argmax(sims)]
        cluster_labels[cluster_id] = articles[best_idx]["title"][:60]

    for i, article in enumerate(articles):
        cid = int(labels[i])
        article["cluster_id"] = cid
        article["cluster_label"] = cluster_labels.get(cid, "unknown")

    log.info(f"Clustered {len(articles)} articles into {n_clusters} narratives")
    return articles