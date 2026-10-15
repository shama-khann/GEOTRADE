import numpy as np
import pandas as pd

def generate_training_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generates synthetic but realistic training data mapping
    GTI features to market outcomes.
    
    Features  → targets:
    gti_score, avg_stress, avg_severity, cluster_density,
    event_type_encoded, region_encoded
    → volatility_spike (bool), bias (bearish/neutral/bullish), risk_level (0-3)
    """
    np.random.seed(42)

    gti_score       = np.random.uniform(0, 100, n_samples)
    avg_stress      = np.clip(gti_score / 100 + np.random.normal(0, 0.15, n_samples), 0, 1)
    avg_severity    = np.clip(gti_score / 100 + np.random.normal(0, 0.12, n_samples), 0, 1)
    cluster_density = np.random.uniform(0, 1, n_samples)
    event_type      = np.random.randint(0, 8, n_samples)   # 8 event types
    region          = np.random.randint(0, 7, n_samples)   # 7 regions

    # Volatility spike: likely when GTI > 65 and stress > 0.6
    vol_prob = (gti_score / 100) * 0.6 + avg_stress * 0.4
    volatility_spike = (vol_prob + np.random.normal(0, 0.1, n_samples) > 0.55).astype(int)

    # Bias: bearish when high stress, bullish when low, neutral middle
    bias_score = avg_stress * 0.5 + (gti_score / 100) * 0.5
    bias = np.where(bias_score > 0.65, 2,       # bearish
           np.where(bias_score < 0.35, 0, 1))   # bullish / neutral

    # Risk level 0-3
    risk = np.floor(gti_score / 25).clip(0, 3).astype(int)

    return pd.DataFrame({
        "gti_score":        gti_score,
        "avg_stress":       avg_stress,
        "avg_severity":     avg_severity,
        "cluster_density":  cluster_density,
        "event_type":       event_type,
        "region":           region,
        "volatility_spike": volatility_spike,
        "bias":             bias,
        "risk_level":       risk,
    })