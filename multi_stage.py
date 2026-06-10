"""
Multi-stage training: expand each game into 3 training samples.

        R1 -> R2  (stage 1)
        R2 -> R3  (stage 2)
        R3 -> R4  (stage 3, R4 = final)

Each sample preserves the final zone label (R4 zone), so the model learns
to predict the final zone from any stage of the game.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from data_io import load_records_csv, save_records_csv
from data_models import CircleRecord, CircleState
from feature_engineering import (
    extract_state_features,
    label_to_index,
    index_to_label,
    rebuild_indices,
    MAP_INDEX,
    ZONE_INDEX,
)
from evaluation import evaluate_records

logger = logging.getLogger(__name__)

COORD_SPACE = 16384


# ---------------------------------------------------------------------------
# zone assignment
# ---------------------------------------------------------------------------

def _pixel_to_zone(x: float, y: float) -> str:
    cx, cy = x / COORD_SPACE, y / COORD_SPACE
    if 0.35 <= cx <= 0.65 and 0.35 <= cy <= 0.65:
        return "center"
    dx, dy = cx - 0.5, cy - 0.5
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    else:
        return "south" if dy > 0 else "north"


# ---------------------------------------------------------------------------
# data expansion
# ---------------------------------------------------------------------------

def expand_full_rings(full_rings_path: str | Path) -> List[CircleRecord]:
    """Read full_rings.json and expand each game into 3 CircleRecords.

    Stage mapping:
        stage 1: rings[0] -> rings[1],  final = rings[3]
        stage 2: rings[1] -> rings[2],  final = rings[3]
        stage 3: rings[2] -> rings[3],  final = rings[3]
    """
    path = Path(full_rings_path)
    games = json.loads(path.read_text(encoding="utf-8"))
    records: List[CircleRecord] = []

    for g in games:
        rings = g["rings"]
        if len(rings) < 4:
            continue

        final_ring = rings[3]
        final_zone = _pixel_to_zone(final_ring["x"], final_ring["y"])

        stages = [(rings[0], rings[1], 1), (rings[1], rings[2], 2), (rings[2], rings[3], 3)]

        for current, next_ring, stage_num in stages:
            records.append(CircleRecord(
                match_id=g["match_id"],
                map_name=g["map_name"],
                stage=stage_num,
                circle_x=current["x"],
                circle_y=current["y"],
                circle_radius=current["r"],
                next_circle_x=next_ring["x"],
                next_circle_y=next_ring["y"],
                next_circle_radius=next_ring["r"],
                final_zone=final_zone,
            ))

    return records


def expand_to_csv(full_rings_path: str | Path, output_path: str | Path) -> Path:
    records = expand_full_rings(full_rings_path)
    path = Path(output_path)
    save_records_csv(records, path)
    logger.info("Expanded %d games -> %d records -> %s",
                len(json.loads(Path(full_rings_path).read_text())),
                len(records), path)
    return path


# ---------------------------------------------------------------------------
# multi-stage predictor (RF on all stages)
# ---------------------------------------------------------------------------

class MultiStageRFPredictor:
    """RF trained on stage-annotated multi-stage data.

    Uses enhanced features including pull vectors and stage-aware features.
    """

    def __init__(self, model: RandomForestClassifier):
        self.model = model

    def predict_proba(self, state: CircleState) -> Dict[str, float]:
        feats = np.array([_extract_enhanced_features(state)], dtype=float)
        probs = self.model.predict_proba(feats)[0]
        mapping = {index_to_label(int(l)): float(p)
                   for l, p in zip(self.model.classes_, probs)}
        for zone in ["north", "south", "east", "west", "center"]:
            mapping.setdefault(zone, 0.0)
        return mapping

    def predict(self, state: CircleState) -> Tuple[str, Dict[str, float]]:
        probs = self.predict_proba(state)
        return max(probs, key=probs.get), probs


def _extract_enhanced_features(state: CircleState) -> List[float]:
    """Enhanced feature vector including pull direction and shrink ratio."""
    base = extract_state_features(state)

    # Pull direction features (normalized relative position)
    cx_norm = state.circle_x / COORD_SPACE
    cy_norm = state.circle_y / COORD_SPACE
    cr_norm = state.circle_radius / COORD_SPACE

    # Distance from map center
    dist_center = np.sqrt((cx_norm - 0.5) ** 2 + (cy_norm - 0.5) ** 2)

    # Ring coverage of map
    coverage = (state.circle_radius / COORD_SPACE) ** 2

    return base + [dist_center, coverage, cx_norm * cy_norm]


@dataclass
class MultiStageResult:
    model: RandomForestClassifier
    sample_count: int
    feature_count: int
    stage_counts: Dict[int, int]
    cv_scores: List[float]


def train_multistage(
    records: List[CircleRecord],
    cv: int = 5,
    random_state: int = 42,
) -> MultiStageResult:
    """Train RF on multi-stage data with enhanced features."""
    if not records:
        raise ValueError("No records provided")

    maps_in_data = sorted({r.map_name for r in records})
    rebuild_indices(maps_in_data, ["north", "south", "east", "west", "center"])

    X = np.array([_extract_enhanced_features(
        CircleState(r.map_name, r.circle_x, r.circle_y, r.circle_radius, r.stage)
    ) for r in records], dtype=float)
    y = np.array([label_to_index(r.final_zone) for r in records], dtype=int)

    stage_counts = {}
    for r in records:
        stage_counts[r.stage] = stage_counts.get(r.stage, 0) + 1

    model = RandomForestClassifier(
        n_estimators=200, random_state=random_state,
        class_weight="balanced", max_depth=10,
    )

    cv_scores = []
    if cv > 1 and len(records) >= cv * 2:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        cv_scores = list(cross_val_score(model, X, y, cv=skf, scoring="accuracy"))

    model.fit(X, y)
    return MultiStageResult(
        model=model,
        sample_count=len(records),
        feature_count=X.shape[1],
        stage_counts=stage_counts,
        cv_scores=cv_scores,
    )


def build_multistage_predictor(model: RandomForestClassifier) -> MultiStageRFPredictor:
    return MultiStageRFPredictor(model)


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_by_stage(records: List[CircleRecord], predictor) -> Dict[int, dict]:
    """Evaluate accuracy broken down by stage."""
    by_stage: Dict[int, list] = {}
    for r in records:
        by_stage.setdefault(r.stage, []).append(r)

    results = {}
    for stage, stage_recs in sorted(by_stage.items()):
        result = evaluate_records(stage_recs, predictor)
        results[stage] = {
            "count": result.total,
            "top1": result.accuracy,
            "top3": result.top3_accuracy,
        }
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Expand data
    records = expand_full_rings("collected_data/full_rings.json")
    print(f"Expanded to {len(records)} records")
    for s in [1, 2, 3]:
        count = sum(1 for r in records if r.stage == s)
        print(f"  Stage {s}: {count} samples")

    # Train
    result = train_multistage(records, cv=5)
    print(f"\nMulti-stage RF:")
    print(f"  Samples: {result.sample_count}, Features: {result.feature_count}")
    print(f"  Stage distribution: {result.stage_counts}")
    if result.cv_scores:
        print(f"  5-Fold CV: {np.mean(result.cv_scores):.3f} (+/- {np.std(result.cv_scores)*2:.3f})")

    # Per-stage evaluation
    predictor = build_multistage_predictor(result.model)
    stage_results = evaluate_by_stage(records, predictor)
    print("\nPer-stage accuracy (training set):")
    for stage, r in stage_results.items():
        print(f"  Stage {stage}: top1={r['top1']:.3f} top3={r['top3']:.3f} (n={r['count']})")
