"""Evaluation routines for rule-based, ML, and hybrid systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.inference_engine import ForwardChainingEngine
from src.ml_model import NetworkMLModel


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


class SystemEvaluator:
    def __init__(self, labels: List[str]):
        self.labels = labels

    @staticmethod
    def _compute_metrics(y_true, y_pred) -> Metrics:
        return Metrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average="weighted", zero_division=0),
            recall=recall_score(y_true, y_pred, average="weighted", zero_division=0),
            f1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
        )

    def evaluate_ml(self, y_true, y_pred) -> Metrics:
        return self._compute_metrics(y_true, y_pred)

    def evaluate_rule_based(self, X_test_raw, y_true, engine: ForwardChainingEngine, label_to_id: Dict[str, int]) -> Metrics:
        preds = []
        for _, row in X_test_raw.iterrows():
            result = engine.infer(row.to_dict())
            diagnosis = result.diagnosis if result.diagnosis in label_to_id else "normal"
            preds.append(label_to_id[diagnosis])
        return self._compute_metrics(y_true, np.array(preds))

    def evaluate_hybrid(
        self,
        X_test_raw,
        X_test_processed,
        y_true,
        engine: ForwardChainingEngine,
        ml_model: NetworkMLModel,
        id_to_label: Dict[int, str],
        label_to_id: Dict[str, int],
        rule_threshold: float = 0.9,
    ) -> Metrics:
        preds = []
        for idx, row in X_test_raw.iterrows():
            rule_result = engine.infer(row.to_dict())
            if rule_result.confidence >= rule_threshold and rule_result.diagnosis in label_to_id:
                preds.append(label_to_id[rule_result.diagnosis])
            else:
                ml_idx = int(ml_model.predict(X_test_processed.loc[[idx]])[0])
                preds.append(ml_idx if ml_idx in id_to_label else label_to_id["normal"])
        return self._compute_metrics(y_true, np.array(preds))
