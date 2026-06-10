"""Model training — Random Forest supervised classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

from data_models import CircleRecord, CircleState, DEFAULT_ZONES
from feature_engineering import (
    extract_record_features,
    extract_state_features,
    label_to_index,
    index_to_label,
    MAP_INDEX,
    ZONE_INDEX,
    rebuild_indices,
)


@dataclass(frozen=True)
class TrainingResult:
    model: RandomForestClassifier
    feature_count: int
    sample_count: int
    cv_scores: Optional[List[float]] = None


class CircleZoneModel:
    """Random Forest predictor — conforms to the Predictor protocol."""

    def __init__(self, model: RandomForestClassifier):
        self.model = model

    def predict_proba(self, state: CircleState) -> dict[str, float]:
        feats = np.array([extract_state_features(state)], dtype=float)
        probs = self.model.predict_proba(feats)[0]
        mapping = {index_to_label(int(l)): float(p) for l, p in zip(self.model.classes_, probs)}
        for zone in DEFAULT_ZONES:
            mapping.setdefault(zone, 0.0)
        return mapping

    def predict(self, state: CircleState) -> tuple[str, dict[str, float]]:
        probs = self.predict_proba(state)
        return max(probs, key=probs.get), probs


def train_model(
    records: Iterable[CircleRecord],
    cv: int = 0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    """Train a RandomForest classifier. If cv > 0, also run stratified K-fold CV."""
    records_list = list(records)
    if not records_list:
        raise ValueError("Cannot train model without records")

    # Rebuild feature indices from actual data
    maps_in_data = sorted({r.map_name for r in records_list})
    zones_in_data = sorted({r.final_zone for r in records_list})
    rebuild_indices(maps_in_data, zones_in_data)

    X = np.array([extract_record_features(r) for r in records_list], dtype=float)
    y = np.array([label_to_index(r.final_zone) for r in records_list], dtype=int)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced",
        max_depth=8,
    )

    cv_scores = None
    if cv > 1 and len(records_list) >= cv * 2:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        cv_scores = list(cross_val_score(model, X, y, cv=skf, scoring="accuracy"))

    model.fit(X, y)
    return TrainingResult(
        model=model,
        feature_count=X.shape[1],
        sample_count=len(records_list),
        cv_scores=cv_scores,
    )


def build_predictor(model: RandomForestClassifier) -> CircleZoneModel:
    return CircleZoneModel(model)
