from __future__ import annotations

import argparse
from pathlib import Path

from data_io import load_records_csv, save_records_csv
from data_models import CircleState
from dataset import build_sample_dataset
from evaluation import evaluate_records
from predictor import make_default_predictor
from training import build_predictor, train_model
from visualization import plot_heatmap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ALGS circle zone prediction project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a machine learning model from CSV data")
    train_parser.add_argument("--data", type=Path, required=False, help="path to a CSV file with training records")

    predict_parser = subparsers.add_parser("predict", help="run a prediction for a single state")
    predict_parser.add_argument("--map-name", required=True)
    predict_parser.add_argument("--x", type=float, required=True)
    predict_parser.add_argument("--y", type=float, required=True)
    predict_parser.add_argument("--radius", type=float, required=True)
    predict_parser.add_argument("--stage", type=int, required=True)
    predict_parser.add_argument("--heatmap", type=Path, help="optional output image path")

    eval_parser = subparsers.add_parser("evaluate", help="evaluate on a CSV dataset or the built-in sample data")
    eval_parser.add_argument("--data", type=Path, required=False)

    export_parser = subparsers.add_parser("export-sample", help="export the sample dataset to CSV")
    export_parser.add_argument("--output", type=Path, required=True)

    return parser


def cmd_train(args: argparse.Namespace) -> None:
    if args.data:
        records = load_records_csv(args.data)
    else:
        records = build_sample_dataset()

    result = train_model(records)
    print("Training complete")
    print(f"  samples: {result.sample_count}")
    print(f"  features: {result.feature_count}")
    print("  model: RandomForestClassifier")


def cmd_predict(args: argparse.Namespace) -> None:
    state = CircleState(args.map_name, args.x, args.y, args.radius, args.stage)
    predictor = make_default_predictor()
    best_zone, probs = predictor.predict(state)
    print(f"Predicted zone: {best_zone}")
    for zone, value in sorted(probs.items(), key=lambda item: item[1], reverse=True):
        print(f"  {zone}: {value:.3f}")
    if args.heatmap:
        plot_heatmap(probs, f"Prediction for {args.map_name}", args.heatmap)
        print(f"Heatmap saved to {args.heatmap}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    records = load_records_csv(args.data) if args.data else build_sample_dataset()
    result = evaluate_records(records, make_default_predictor())
    print("Evaluation results")
    print(f"  records: {result.total}")
    print(f"  top-1 accuracy: {result.accuracy:.3f}")
    print(f"  top-3 accuracy: {result.top3_accuracy:.3f}")


def cmd_export_sample(args: argparse.Namespace) -> None:
    save_records_csv(build_sample_dataset(), args.output)
    print(f"Sample dataset exported to {args.output}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "export-sample":
        cmd_export_sample(args)


if __name__ == "__main__":
    main()
