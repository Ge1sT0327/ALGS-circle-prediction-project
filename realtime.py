"""
Real-time ALGS circle prediction tool.

Usage modes:
  1. Interactive CLI — enter ring data manually during a game
  2. Quick predict — one-shot prediction from command line
  3. Simulate — replay a full game ring sequence step by step

Models: lstm, transformer, rf (multi-stage), gmm, baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from data_models import CircleState, DEFAULT_ZONES
from data_io import load_records_csv
from predictor import SimpleCirclePredictor
from multi_stage import expand_full_rings, train_multistage, build_multistage_predictor
from unsupervised import train_gmm, train_kde
from visualization import plot_heatmap


# ---------------------------------------------------------------------------
# prediction engine
# ---------------------------------------------------------------------------

class PredictionEngine:
    """Loads all models and provides a unified prediction interface."""

    def __init__(self, full_rings_path: str | Path | None = None,
                 lstm_result = None, transformer_result = None):
        self.models: Dict[str, object] = {}
        self.ring_history: List[dict] = []
        self.map_name: str = "worlds_edge"

        # Fast models (no training data needed)
        self.models["baseline"] = SimpleCirclePredictor()

        # Trainable models
        if full_rings_path:
            path = Path(full_rings_path)
            if path.exists():
                records = expand_full_rings(path)
                csv_path = Path("collected_data/rings_multistage.csv")
                from data_io import save_records_csv
                save_records_csv(records, csv_path)

                result = train_multistage(records, cv=0)
                self.models["rf"] = build_multistage_predictor(result.model)
                self.models["gmm"] = train_gmm(records)
                self.models["kde"] = train_kde(records)
                self._records = records

        # Sequential models
        if lstm_result is not None:
            from sequential import SequentialPredictor
            self.models["lstm"] = SequentialPredictor(lstm_result)
        if transformer_result is not None:
            from sequential import SequentialPredictor
            self.models["transformer"] = SequentialPredictor(transformer_result)

    def add_ring(self, x: float, y: float, r: float) -> None:
        self.ring_history.append({"x": x, "y": y, "r": r})

    def predict(self, model: str) -> Dict[str, float]:
        if model not in self.models:
            raise ValueError(f"Model '{model}' not available. Choose: {list(self.models.keys())}")

        predictor = self.models[model]
        n_rings = len(self.ring_history)

        if model in ("lstm", "transformer"):
            if n_rings < 1:
                raise ValueError("Need at least 1 ring for sequential model")
            best, probs = predictor.predict_sequence(self.ring_history, self.map_name)
        else:
            if n_rings < 1:
                raise ValueError("No ring data available")
            latest = self.ring_history[-1]
            stage = min(n_rings, 5)
            state = CircleState(self.map_name, latest["x"], latest["y"], latest["r"], stage)
            best, probs = predictor.predict(state)

        return probs

    def predict_best(self, model: str) -> Tuple[str, Dict[str, float]]:
        probs = self.predict(model)
        best = max(probs, key=probs.get)
        return best, probs


# ---------------------------------------------------------------------------
# interactive CLI
# ---------------------------------------------------------------------------

MAP_CHOICES = ["worlds_edge", "storm_point", "e_district"]
MODEL_CHOICES = ["baseline", "rf", "gmm", "kde", "lstm", "transformer"]


def _fmt_probs(probs: Dict[str, float]) -> str:
    lines = []
    for zone in sorted(probs, key=probs.get, reverse=True):
        bar = "#" * int(probs[zone] * 50)
        lines.append(f"  {zone:>8s}: {probs[zone]:.3f} |{bar}")
    return "\n".join(lines)


def interactive_mode(engine: PredictionEngine) -> None:
    """Interactive ring entry during a live match."""
    print("\n" + "=" * 60)
    print("  ALGS Real-Time Circle Predictor")
    print("=" * 60)
    print("\nEnter ring data as:  x y radius")
    print("Commands: map <name>, model <name>, predict, reset, quit")
    print(f"Maps: {MAP_CHOICES}")
    print(f"Models: {[m for m in MODEL_CHOICES if m in engine.models]}")
    print()

    current_model = "rf" if "rf" in engine.models else "baseline"

    while True:
        try:
            cmd = input(f"[{engine.map_name}:{len(engine.ring_history)} rings, {current_model}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cmd:
            continue

        parts = cmd.split()

        if parts[0] == "quit" or parts[0] == "q":
            break

        elif parts[0] == "map" and len(parts) > 1:
            if parts[1] in MAP_CHOICES:
                engine.map_name = parts[1]
                engine.ring_history = []
                print(f"  Map set to {parts[1]}, ring history cleared.")
            else:
                print(f"  Unknown map. Choose: {MAP_CHOICES}")

        elif parts[0] == "model" and len(parts) > 1:
            if parts[1] in engine.models:
                current_model = parts[1]
                print(f"  Model set to {parts[1]}.")
            else:
                available = list(engine.models.keys())
                print(f"  Unknown model. Available: {available}")

        elif parts[0] == "reset":
            engine.ring_history = []
            print("  Ring history cleared.")

        elif parts[0] == "predict":
            if not engine.ring_history:
                print("  No ring data. Enter at least one ring first.")
                continue
            try:
                probs = engine.predict(current_model)
                best = max(probs, key=probs.get)
                print(f"\n  >>> Predicted final zone: {best.upper()} <<<")
                print(_fmt_probs(probs))
                print()
            except Exception as e:
                print(f"  Error: {e}")

        elif parts[0] == "all":
            if not engine.ring_history:
                print("  No ring data.")
                continue
            print()
            for m in engine.models:
                try:
                    probs = engine.predict(m)
                    best = max(probs, key=probs.get)
                    print(f"  [{m.upper():>12s}] -> {best:>8s} | " +
                          " | ".join(f"{z[:1]}:{probs[z]:.2f}" for z in DEFAULT_ZONES))
                except Exception:
                    print(f"  [{m.upper():>12s}] -> (error)")
            print()

        else:
            # Try parsing as ring data: x y radius
            try:
                vals = [float(p) for p in parts[:3]]
                x, y, r = vals[0], vals[1], vals[2]
                engine.add_ring(x, y, r)
                print(f"  Ring #{len(engine.ring_history)}: ({x:.0f}, {y:.0f}) r={r:.0f}")

                # Auto-predict after adding a ring
                if len(engine.ring_history) >= 1:
                    probs = engine.predict(current_model)
                    best = max(probs, key=probs.get)
                    print(f"  -> {best.upper()} | " +
                          " | ".join(f"{z}:{probs[z]:.2f}" for z in DEFAULT_ZONES))
            except ValueError:
                print(f"  Unknown command: {cmd}")
                print("  Usage: <x> <y> <radius>  |  map <name>  |  model <name>  |  predict  |  all  |  reset  |  quit")


def simulate_game(engine: PredictionEngine, game_index: int = 0) -> None:
    """Simulate a full game from the dataset, predicting at each stage."""
    path = Path("collected_data/full_rings.json")
    if not path.exists():
        print("No full_rings.json found.")
        return

    games = json.loads(path.read_text(encoding="utf-8"))
    if game_index >= len(games):
        game_index = 0

    game = games[game_index]
    engine.map_name = game["map_name"]
    engine.ring_history = []

    print(f"\n=== Simulating: {game['game_desc']} ===")
    print(f"    Map: {game['map_name']}  |  ID: {game['match_id'][:12]}")
    print()

    for i, ring in enumerate(game["rings"]):
        engine.add_ring(ring["x"], ring["y"], ring["r"])
        stage_name = f"Round {i+1}" if i < 3 else "FINAL"
        print(f"--- {stage_name}: ({ring['x']:.0f}, {ring['y']:.0f}) r={ring['r']:.0f} ---")

        if i < 3:  # Predict before final
            print(f"{'Model':>12s}  {'Pred':>8s}  " + "  ".join(f"{z:>6s}" for z in DEFAULT_ZONES))
            print("-" * 65)
            for m in engine.models:
                try:
                    probs = engine.predict(m)
                    best = max(probs, key=probs.get)
                    parts = "  ".join(f"{probs[z]:.3f}" for z in DEFAULT_ZONES)
                    marker = " ✓" if best == _ring_zone(game["rings"][3]) else ""
                    print(f"{m:>12s}  {best:>8s}  {parts}{marker}")
                except Exception:
                    print(f"{m:>12s}  {'error':>8s}")
            print()

    final_zone = _ring_zone(game["rings"][3])
    print(f"Actual final zone: {final_zone.upper()}")
    print(f"Final ring: ({game['rings'][3]['x']:.0f}, {game['rings'][3]['y']:.0f}) r={game['rings'][3]['r']:.0f}")


def _ring_zone(final_ring: dict) -> str:
    from multi_stage import _pixel_to_zone
    return _pixel_to_zone(final_ring["x"], final_ring["y"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ALGS Real-Time Circle Predictor")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Interactive
    sub.add_parser("interactive", help="Interactive mode (enter rings manually)")

    # Simulate
    sim_p = sub.add_parser("simulate", help="Simulate a full game from dataset")
    sim_p.add_argument("--game", type=int, default=0, help="Game index (0-459)")

    # Quick predict
    qp = sub.add_parser("quick", help="One-shot prediction")
    qp.add_argument("--map", required=True, choices=MAP_CHOICES)
    qp.add_argument("--x", type=float, required=True)
    qp.add_argument("--y", type=float, required=True)
    qp.add_argument("--radius", type=float, required=True)
    qp.add_argument("--stage", type=int, default=1)
    qp.add_argument("--model", default="rf", choices=MODEL_CHOICES)
    qp.add_argument("--heatmap", type=Path)

    args = parser.parse_args()

    # Load engine with all models
    full_rings = Path("collected_data/full_rings.json")
    engine = PredictionEngine(full_rings if full_rings.exists() else None)

    if args.mode == "interactive":
        interactive_mode(engine)

    elif args.mode == "simulate":
        # Try loading sequential models
        try:
            from sequential import SequentialPredictor, train_sequential
            print("Training LSTM (this will take a moment)...")
            lstm_result = train_sequential("collected_data/full_rings.json",
                                           model_type="lstm", epochs=50)
            engine.models["lstm"] = SequentialPredictor(lstm_result)
            print(f"LSTM val_acc: {lstm_result.val_accuracy:.3f}")
        except Exception as e:
            print(f"LSTM skipped: {e}")

        simulate_game(engine, args.game)

    elif args.mode == "quick":
        engine.map_name = args.map
        engine.add_ring(args.x, args.y, args.radius)

        if args.model not in engine.models:
            print(f"Model '{args.model}' not available. Train it first or use: {list(engine.models.keys())}")
            sys.exit(1)

        best, probs = engine.predict_best(args.model)

        print(f"Map: {args.map}  |  Ring: ({args.x:.0f}, {args.y:.0f}) r={args.radius:.0f}  |  Stage: {args.stage}")
        print(f"Model: {args.model}")
        print(f"Predicted zone: {best.upper()}")
        print(_fmt_probs(probs))

        if args.heatmap:
            plot_heatmap(probs, f"{args.model} — {args.map} stage {args.stage}", args.heatmap)
            print(f"Heatmap saved: {args.heatmap}")


if __name__ == "__main__":
    main()
