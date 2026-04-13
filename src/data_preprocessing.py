"""Data loading and preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


@dataclass
class DataBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: Pipeline
    label_encoder: LabelEncoder
    feature_names: List[str]


class DataProcessor:
    """Handle loading, cleaning, scaling, and splitting of network metrics."""

    FEATURE_COLUMNS = [
        "latency",
        "packet_loss",
        "jitter",
        "bandwidth_usage",
        "congestion",
        "traffic_anomaly",
    ]

    LABEL_COLUMN = "label"

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def load_csv(self, csv_path: str | Path) -> pd.DataFrame:
        return pd.read_csv(csv_path)

    def prepare(self, df: pd.DataFrame) -> DataBundle:
        data = df.copy()

        missing_cols = set(self.FEATURE_COLUMNS + [self.LABEL_COLUMN]) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {sorted(missing_cols)}")

        X = data[self.FEATURE_COLUMNS]
        y_raw = data[self.LABEL_COLUMN]

        preprocessor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        X_processed = preprocessor.fit_transform(X)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_raw)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed,
            y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded,
        )

        return DataBundle(
            X_train=pd.DataFrame(X_train, columns=self.FEATURE_COLUMNS),
            X_test=pd.DataFrame(X_test, columns=self.FEATURE_COLUMNS),
            y_train=pd.Series(y_train),
            y_test=pd.Series(y_test),
            preprocessor=preprocessor,
            label_encoder=label_encoder,
            feature_names=list(self.FEATURE_COLUMNS),
        )
