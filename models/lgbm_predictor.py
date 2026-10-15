import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import os
import logging

from models.training_data import generate_training_data
from models.feature_builder import BIAS_LABELS, RISK_LABELS

log = logging.getLogger(__name__)

MODEL_DIR  = "models/saved"
VOL_PATH   = f"{MODEL_DIR}/volatility.pkl"
BIAS_PATH  = f"{MODEL_DIR}/bias.pkl"
RISK_PATH  = f"{MODEL_DIR}/risk.pkl"

FEATURES = ["gti_score", "avg_stress", "avg_severity",
            "cluster_density", "event_type", "region"]


def _train_models():
    """Train all three models on synthetic data and save them."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    log.info("Training LightGBM models on synthetic data...")

    df = generate_training_data(2000)
    X  = df[FEATURES]

    params = {
        "n_estimators":    200,
        "learning_rate":   0.05,
        "max_depth":       4,
        "num_leaves":      15,
        "random_state":    42,
        "verbose":         -1,
    }

    # Volatility spike (binary)
    vol_model = lgb.LGBMClassifier(**params)
    vol_model.fit(X, df["volatility_spike"])
    with open(VOL_PATH, "wb") as f:
        pickle.dump(vol_model, f)

    # Directional bias (3-class)
    bias_model = lgb.LGBMClassifier(**params)
    bias_model.fit(X, df["bias"])
    with open(BIAS_PATH, "wb") as f:
        pickle.dump(bias_model, f)

    # Risk level (4-class)
    risk_model = lgb.LGBMClassifier(**params)
    risk_model.fit(X, df["risk_level"])
    with open(RISK_PATH, "wb") as f:
        pickle.dump(risk_model, f)

    log.info("Models trained and saved.")
    return vol_model, bias_model, risk_model


def load_models():
    """Load saved models, or train fresh ones if not found."""
    if all(os.path.exists(p) for p in [VOL_PATH, BIAS_PATH, RISK_PATH]):
        log.info("Loading saved LightGBM models...")
        with open(VOL_PATH,  "rb") as f: vol_model  = pickle.load(f)
        with open(BIAS_PATH, "rb") as f: bias_model = pickle.load(f)
        with open(RISK_PATH, "rb") as f: risk_model = pickle.load(f)
    else:
        vol_model, bias_model, risk_model = _train_models()

    return vol_model, bias_model, risk_model


def predict(feature_vector: dict) -> dict:
    """
    Takes a feature vector dict, returns market predictions.
    """
    vol_model, bias_model, risk_model = load_models()

    X = pd.DataFrame([feature_vector])[FEATURES]

    vol_prob   = vol_model.predict_proba(X)[0][1]
    bias_class = int(bias_model.predict(X)[0])
    risk_class = int(risk_model.predict(X)[0])

    return {
        "volatility_spike_probability": round(float(vol_prob), 4),
        "volatility_spike_expected":    vol_prob > 0.5,
        "directional_bias":             BIAS_LABELS[bias_class],
        "risk_level":                   RISK_LABELS[risk_class],
    }