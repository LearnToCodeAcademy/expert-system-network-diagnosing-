"""CLI interface for training and running the hybrid expert system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from src.data_preprocessing import DataProcessor
from src.evaluator import SystemEvaluator
from src.inference_engine import ForwardChainingEngine
from src.ml_model import NetworkMLModel
from src.rules import RuleKnowledgeBase
from src.utils import HybridDiagnosisService, ensure_dirs

MODEL_PATH = Path("models/trained_model.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")


def train_pipeline(dataset_path: str) -> None:
    ensure_dirs()
    processor = DataProcessor()
    df = processor.load_csv(dataset_path)
    bundle = processor.prepare(df)

    model = NetworkMLModel()
    model.train(bundle.X_train, bundle.y_train)
    model.save(MODEL_PATH)

    joblib.dump(bundle.preprocessor, PREPROCESSOR_PATH)
    joblib.dump(bundle.label_encoder, ENCODER_PATH)

    engine = ForwardChainingEngine(RuleKnowledgeBase.default())

    evaluator = SystemEvaluator(labels=bundle.label_encoder.classes_.tolist())
    ml_pred = model.predict(bundle.X_test)
    ml_metrics = evaluator.evaluate_ml(bundle.y_test, ml_pred).as_dict()

    raw_test = df.loc[bundle.X_test.index, DataProcessor.FEATURE_COLUMNS].reset_index(drop=True)
    y_test = bundle.y_test.reset_index(drop=True)
    x_test_processed = bundle.X_test.reset_index(drop=True)

    label_to_id = {label: idx for idx, label in enumerate(bundle.label_encoder.classes_)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    rule_metrics = evaluator.evaluate_rule_based(raw_test, y_test, engine, label_to_id).as_dict()
    hybrid_metrics = evaluator.evaluate_hybrid(
        raw_test,
        x_test_processed,
        y_test,
        engine,
        model,
        id_to_label,
        label_to_id,
        rule_threshold=0.9,
    ).as_dict()

    report = {
        "ml": ml_metrics,
        "rule_based": rule_metrics,
        "hybrid": hybrid_metrics,
    }
    print("Training complete. Evaluation metrics:")
    print(json.dumps(report, indent=2))


def load_service() -> HybridDiagnosisService:
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists() or not ENCODER_PATH.exists():
        raise FileNotFoundError("Model assets not found. Run `python app/main.py train --data <csv>` first.")

    model = NetworkMLModel()
    model.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    engine = ForwardChainingEngine(RuleKnowledgeBase.default())

    return HybridDiagnosisService(engine, model, preprocessor, label_encoder)


def diagnose_from_args(args) -> None:
    service = load_service()
    symptoms = {
        "latency": args.latency,
        "packet_loss": args.packet_loss,
        "jitter": args.jitter,
        "bandwidth_usage": args.bandwidth_usage,
        "congestion": args.congestion,
        "traffic_anomaly": args.traffic_anomaly,
    }
    result = service.diagnose(symptoms)
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Network troubleshooting expert system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train ML model and evaluate system")
    train_parser.add_argument("--data", required=True, help="Path to training CSV dataset")

    diagnose_parser = subparsers.add_parser("diagnose", help="Diagnose network issue from symptoms")
    diagnose_parser.add_argument("--latency", type=float, required=True)
    diagnose_parser.add_argument("--packet_loss", type=float, required=True)
    diagnose_parser.add_argument("--jitter", type=float, required=True)
    diagnose_parser.add_argument("--bandwidth_usage", type=float, required=True)
    diagnose_parser.add_argument("--congestion", type=float, required=True)
    diagnose_parser.add_argument("--traffic_anomaly", type=float, required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_pipeline(args.data)
    elif args.command == "diagnose":
        diagnose_from_args(args)


if __name__ == "__main__":
    main()
