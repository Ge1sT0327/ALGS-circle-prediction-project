from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from data_models import CircleRecord, CircleState, DEFAULT_ZONES
from feature_engineering import extract_record_features, label_to_index, index_to_label


@dataclass(frozen=True)
class TrainingResult:
    model: RandomForestClassifier
    feature_count: int
    sample_count: int


class CircleZoneModel:
    def __init__(self, model: RandomForestClassifier) -> None:
        self.model = model

    def predict_proba(self, state: CircleState) -> dict[str, float]:
        from feature_engineering import extract_state_features

        features = np.array([extract_state_features(state)], dtype=float)
        probabilities = self.model.predict_proba(features)[0]
        labels = list(self.model.classes_)
        mapping = {index_to_label(int(label)): float(prob) for label, prob in zip(labels, probabilities)}
        for zone in DEFAULT_ZONES:
            mapping.setdefault(zone, 0.0)
        return mapping

    def predict(self, state: CircleState) -> tuple[str, dict[str, float]]:
        probs = self.predict_proba(state)
        best = max(probs, key=probs.get)
        return best, probs


def train_model(records: Iterable[CircleRecord]) -> TrainingResult:
    records = list(records)
    if not records:
        raise ValueError("Cannot train model without records")

    x = np.array([extract_record_features(record) for record in records], dtype=float)
    y = np.array([label_to_index(record.final_zone) for record in records], dtype=int)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_depth=8,
    )
    model.fit(x, y)
    return TrainingResult(model=model, feature_count=x.shape[1], sample_count=len(records))


def build_predictor(model: RandomForestClassifier) -> CircleZoneModel:
    return CircleZoneModel(model)
