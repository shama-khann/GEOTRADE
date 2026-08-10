import json
import asyncio
import logging
import os
import threading
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from ingestion.queue_writer import get_redis_client
from nlp.classifier import classify_event
from nlp.sentiment import analyze_sentiment
from nlp.clustering import cluster_articles
from nlp.gti_aggregator import compute_gti
from models.feature_builder import build_feature_vector
from models.lgbm_predictor import predict, load_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

QUEUE_NAME = os.getenv("REDIS_QUEUE_NAME", "gti:articles")
BATCH_SIZE = 20

# Shared state — updated by background worker, read by API
latest_state = {
    "gti_score": 0,
    "volatility": "Normal",
    "volatility_prob": 0,
    "bias": "Neutral",
    "risk_level": "Low",
    "stress": 0,
    "severity": 0,
    "narratives": [],
    "articles": [],
    "region_scores": {
        "north_america": 0, "south_america": 0, "europe": 0,
        "africa": 0, "middle_east": 0, "asia": 0, "oceania": 0
    },
    "last_updated": None
}

connected_clients: list[WebSocket] = []


def compute_region_scores(articles: list[dict]) -> dict:
    region_map = {
        "global":      ["north_america","europe","asia","africa","south_america","oceania","middle_east"],
        "middle_east": ["middle_east"],
        "europe":      ["europe"],
        "asia":        ["asia"],
        "africa":      ["africa"],
        "americas":    ["north_america","south_america"],
        "oceania":     ["oceania"],
    }
    scores = {r: [] for r in ["north_america","south_america","europe","africa","middle_east","asia","oceania"]}
    for article in articles:
        region = article.get("region", "global")
        stress = article.get("stress_score", 0.5)
        for mapped in region_map.get(region, ["north_america"]):
            scores[mapped].append(stress)
    return {r: round(sum(v)/len(v)*100, 1) if v else 0 for r, v in scores.items()}


def pipeline_worker():
    redis_client = get_redis_client()
    load_models()
    log.info("Pipeline worker ready.")
    batch = []

    while True:
        try:
            item = redis_client.brpop(QUEUE_NAME, timeout=5)
            if item:
                _, raw = item
                article = json.loads(raw)
                batch.append(article)

            if len(batch) >= BATCH_SIZE or (len(batch) > 0 and item is None):
                log.info(f"Processing batch of {len(batch)}")
                enriched = []
                for a in batch:
                    clf  = classify_event(a["title"], a["summary"])
                    sent = analyze_sentiment(a["title"], a["summary"])
                    enriched.append({**a, **clf, **sent})

                enriched   = cluster_articles(enriched)
                gti_result = compute_gti(enriched)
                features   = build_feature_vector(gti_result, enriched)
                preds      = predict(features)
                region_scores = compute_region_scores(enriched)

                latest_state.update({
                    "gti_score":        gti_result["gti_score"],
                    "volatility":       "Spike Expected" if preds["volatility_spike_expected"] else "Normal",
                    "volatility_prob":  round(preds["volatility_spike_probability"] * 100),
                    "bias":             preds["directional_bias"],
                    "risk_level":       preds["risk_level"],
                    "stress":           gti_result["breakdown"]["avg_stress_score"],
                    "severity":         gti_result["breakdown"]["avg_event_severity"],
                    "narratives":       gti_result["top_narratives"],
                    "region_scores":    region_scores,
                    "articles":         [
                        {"title": a["title"], "source": a["source"],
                         "region": a["region"], "stress_score": a.get("stress_score",0),
                         "event_type": a.get("event_type",""), "url": a.get("url","")}
                        for a in enriched[:30]
                    ],
                    "last_updated": datetime.now(timezone.utc).isoformat()
                })
                batch = []

        except Exception as e:
            log.error(f"Pipeline error: {e}")
            batch = []


@app.on_event("startup")
def startup():
    t = threading.Thread(target=pipeline_worker, daemon=True)
    t.start()


@app.get("/state")
def get_state():
    return latest_state


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    try:
        await ws.send_json(latest_state)
        while True:
            await asyncio.sleep(10)
            await ws.send_json(latest_state)
    except WebSocketDisconnect:
        connected_clients.remove(ws)


@app.get("/")
def root():
    with open("frontend.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())