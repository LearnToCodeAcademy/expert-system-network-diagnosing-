"""Feature engineering helpers for network metrics."""

from __future__ import annotations

import pandas as pd


class FeatureEngineer:
    """Create derived signals used by downstream models."""

    @staticmethod
    def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["delay_loss_interaction"] = out["latency"] * (out["packet_loss"] + 1.0)
        out["stability_index"] = 1.0 / (1.0 + out["jitter"] + out["packet_loss"])
        out["utilization_pressure"] = out["bandwidth_usage"] * out["congestion"]
        return out
