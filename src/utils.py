"""Utility functions and hybrid decision logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.inference_engine import ForwardChainingEngine
from src.ml_model import NetworkMLModel


class HybridDiagnosisService:
    """Blend rule-based reasoning with ML fallback."""

    def __init__(
        self,
        engine: ForwardChainingEngine,
        ml_model: NetworkMLModel,
        preprocessor,
        label_encoder,
        rule_threshold: float = 0.9,
    ):
        self.engine = engine
        self.ml_model = ml_model
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.rule_threshold = rule_threshold

    def diagnose(self, symptoms: Dict[str, Any]) -> Dict[str, Any]:
        rule_result = self.engine.infer(symptoms)
        if rule_result.confidence >= self.rule_threshold and rule_result.diagnosis != "unknown":
            return {
                "diagnosis": rule_result.diagnosis,
                "confidence": round(rule_result.confidence, 4),
                "source": "rule-based",
                "recommendation": rule_result.recommendation,
                "explanation": rule_result.explanation,
            }

        row = pd.DataFrame([symptoms])
        processed = self.preprocessor.transform(row)
        ml_pred = self.ml_model.predict_single(processed, self.label_encoder.classes_.tolist())

        recommendation = (
            "Model suggested this diagnosis. Verify with additional telemetry before remediation."
        )
        return {
            "diagnosis": ml_pred["diagnosis"],
            "confidence": round(ml_pred["confidence"], 4),
            "source": "ml",
            "recommendation": recommendation,
            "explanation": f"Rule confidence {rule_result.confidence:.2f} below threshold {self.rule_threshold:.2f}.",
        }


def ensure_dirs() -> None:
    for path in ["data/raw", "data/processed", "models"]:
        Path(path).mkdir(parents=True, exist_ok=True)
