"""Optional Streamlit dashboard for interactive diagnosis."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import streamlit as st

from src.inference_engine import ForwardChainingEngine
from src.ml_model import NetworkMLModel
from src.rules import RuleKnowledgeBase
from src.utils import HybridDiagnosisService

MODEL_PATH = "models/trained_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


@st.cache_resource
def load_service() -> HybridDiagnosisService:
    model = NetworkMLModel()
    model.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    encoder = joblib.load(ENCODER_PATH)
    engine = ForwardChainingEngine(RuleKnowledgeBase.default())
    return HybridDiagnosisService(engine, model, preprocessor, encoder)


def main() -> None:
    st.set_page_config(page_title="Network Expert System", layout="centered")
    st.title("Rule-Based Expert System for Network Troubleshooting")
    st.caption("Hybrid diagnosis: rule engine + machine learning fallback")

    latency = st.slider("Latency (ms)", 0.0, 300.0, 60.0)
    packet_loss = st.slider("Packet Loss (%)", 0.0, 20.0, 1.0)
    jitter = st.slider("Jitter (ms)", 0.0, 80.0, 10.0)
    bandwidth_usage = st.slider("Bandwidth Usage (%)", 0.0, 100.0, 55.0)
    congestion = st.slider("Congestion (0-1)", 0.0, 1.0, 0.3)
    traffic_anomaly = st.slider("Traffic Anomaly (0-1)", 0.0, 1.0, 0.2)

    if st.button("Diagnose"):
        service = load_service()
        symptoms = {
            "latency": latency,
            "packet_loss": packet_loss,
            "jitter": jitter,
            "bandwidth_usage": bandwidth_usage,
            "congestion": congestion,
            "traffic_anomaly": traffic_anomaly,
        }
        result = service.diagnose(symptoms)

        st.subheader("Diagnosis Result")
        st.json(result)


if __name__ == "__main__":
    main()
