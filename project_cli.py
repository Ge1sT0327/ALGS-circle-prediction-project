"""CLI for ALGS Circle Zone Prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_io import load_records_csv, save_records_csv
from data_models import CircleState
from dataset import build_sample_dataset
from evaluation import evaluate_records
from predictor import make_default_predictor
from training import train_model, build_predictor
from unsupervised import train_gmm, train_kde, train_kmeans
from visualization import plot_heatmap


MODEL_CHOICES = ("baseline", "rf", "gmm", "kde")


def _load_data(path: Path | None):
    if path:
        records = load_records_csv(path)
        print(f"Loaded {len(records)} records from {path}")
        return records
    records = build_sample_dataset()
    print(f"Using built-in sample ({len(records)} records)")
    return records


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    records = _load_data(args.data)

    if args.model == "baseline":
        print("Baseline model requires no training.")
        return

    if args.model == "rf":
        result = train_model(records, cv=5)
        print("Training complete — Random Forest")
        print(f"  samples:  {result.sample_count}")
        print(f"  features: {result.feature_count}")
        if result.cv_scores:
            cv_mean = sum(result.cv_scores) / len(result.cv_scores)
            print(f"  5-Fold CV accuracy: {cv_mean:.3f}  (folds: {[f'{s:.3f}' for s in result.cv_scores]})")

    elif args.model == "gmm":
        predictor = train_gmm(records)
        print(f"Training complete — GMM ({len(predictor.map_gmms)} maps)")

    elif args.model == "kde":
        predictor = train_kde(records)
        print(f"Training complete — KDE ({len(predictor.map_kdes)} maps)")

    elif args.model == "all":
        print("=== Random Forest ===")
        result = train_model(records, cv=5)
        print(f"  samples: {result.sample_count}, features: {result.feature_count}")
        if result.cv_scores:
            print(f"  5-Fold CV: {sum(result.cv_scores)/len(result.cv_scores):.3f}")

        print("\n=== GMM ===")
        train_gmm(records)
        print("  done")

        print("\n=== KDE ===")
        train_kde(records)
        print("  done")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def _build_predictor(model: str, records):
    if model == "baseline":
        return make_default_predictor()
    elif model == "rf":
        result = train_model(records)
        return build_predictor(result.model)
    elif model == "gmm":
        return train_gmm(records)
    elif model == "kde":
        return train_kde(records)
    raise ValueError(f"Unknown model: {model}")


def cmd_predict(args: argparse.Namespace) -> None:
    records = _load_data(args.data) if args.data else build_sample_dataset()
    state = CircleState(args.map_name, args.x, args.y, args.radius, args.stage)

    predictor = _build_predictor(args.model, records)
    best_zone, probs = predictor.predict(state)

    print(f"Model:     {args.model}")
    print(f"Map:       {args.map_name}")
    print(f"Circle:    ({args.x:.0f}, {args.y:.0f}) r={args.radius:.0f}  stage={args.stage}")
    print(f"Predicted: {best_zone}")
    print("Probabilities:")
    for zone, val in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "#" * int(val * 40)
        print(f"  {zone:>8s}: {val:.3f}  {bar}")

    if args.heatmap:
        title = f"{args.model.upper()} — {args.map_name} (stage {args.stage})"
        plot_heatmap(probs, title, args.heatmap)
        print(f"Heatmap saved to {args.heatmap}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> None:
    records = _load_data(args.data) if args.data else build_sample_dataset()

    models = ["baseline", "rf", "gmm", "kde"] if args.model == "all" else [args.model]

    for model_name in models:
        predictor = _build_predictor(model_name, records)
        result = evaluate_records(records, predictor)
        print(f"\n{'='*50}")
        print(f"Model: {model_name.upper()}")
        print(f"  Records:    {result.total}")
        print(f"  Top-1 acc:  {result.accuracy:.3f}  ({result.accuracy*100:.1f}%)")
        print(f"  Top-3 acc:  {result.top3_accuracy:.3f}  ({result.top3_accuracy*100:.1f}%)")


# ---------------------------------------------------------------------------
# analyze (K-Means archetypes)
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    records = _load_data(args.data)
    kmeans = train_kmeans(records, n_clusters=args.clusters)

    for map_name in sorted(kmeans.map_models.keys()):
        print(f"\n{'='*60}")
        print(f"Map: {map_name}")
        print(f"{'='*60}")
        for d in kmeans.describe(map_name):
            print(
                f"  Cluster {d['cluster']}: {d['count']:4d} pulls  |  "
                f"dx={d['dx_pixels']:+6d}  dy={d['dy_pixels']:+6d} px  |  "
                f"dist={d['distance_pixels']:5d} px  |  dir={d['direction']}"
            )


# ---------------------------------------------------------------------------
# export-sample
# ---------------------------------------------------------------------------

def cmd_export_sample(args: argparse.Namespace) -> None:
    save_records_csv(build_sample_dataset(), args.output)
    print(f"Sample dataset exported to {args.output}")


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ALGS Circle Zone Prediction")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    train_p = sub.add_parser("train", help="Train a model")
    train_p.add_argument("--data", type=Path, help="CSV file with training records")
    train_p.add_argument("--model", choices=MODEL_CHOICES + ("all",),
                         default="rf", help="Model type (default: rf)")

    # ---- predict ----
    pred_p = sub.add_parser("predict", help="Predict zone for a circle state")
    pred_p.add_argument("--map-name", required=True)
    pred_p.add_argument("--x", type=float, required=True)
    pred_p.add_argument("--y", type=float, required=True)
    pred_p.add_argument("--radius", type=float, required=True)
    pred_p.add_argument("--stage", type=int, required=True)
    pred_p.add_argument("--model", choices=MODEL_CHOICES, default="rf")
    pred_p.add_argument("--data", type=Path, help="CSV to train on (optional)")
    pred_p.add_argument("--heatmap", type=Path, help="Output heatmap image path")

    # ---- evaluate ----
    eval_p = sub.add_parser("evaluate", help="Evaluate model accuracy")
    eval_p.add_argument("--data", type=Path, help="CSV dataset")
    eval_p.add_argument("--model", choices=MODEL_CHOICES + ("all",),
                        default="all", help="Model(s) to evaluate")

    # ---- analyze ----
    ana_p = sub.add_parser("analyze", help="K-Means ring-pull archetype analysis")
    ana_p.add_argument("--data", type=Path, required=True, help="CSV dataset")
    ana_p.add_argument("--clusters", type=int, default=5, help="Number of clusters per map")

    # ---- export-sample ----
    exp_p = sub.add_parser("export-sample", help="Export built-in sample to CSV")
    exp_p.add_argument("--output", type=Path, required=True)

    return parser


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cmd = args.command
    if cmd == "train":
        cmd_train(args)
    elif cmd == "predict":
        cmd_predict(args)
    elif cmd == "evaluate":
        cmd_evaluate(args)
    elif cmd == "analyze":
        cmd_analyze(args)
    elif cmd == "export-sample":
        cmd_export_sample(args)


if __name__ == "__main__":
    main()
