from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from data_models import CircleRecord, CircleState
from predictor import SimpleCirclePredictor


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    top3_accuracy: float
    total: int


def evaluate_records(records: Iterable[CircleRecord], predictor: SimpleCirclePredictor) -> EvaluationResult:
    total = 0
    hits = 0
    top3_hits = 0

    for record in records:
        total += 1
        state = CircleState(
            map_name=record.map_name,
            circle_x=record.circle_x,
            circle_y=record.circle_y,
            circle_radius=record.circle_radius,
            stage=record.stage,
        )
        predicted_zone, scores = predictor.predict(state)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top3 = [zone for zone, _ in ranked[:3]]

        if predicted_zone == record.final_zone:
            hits += 1
        if record.final_zone in top3:
            top3_hits += 1

    if total == 0:
        return EvaluationResult(0.0, 0.0, 0)

    return EvaluationResult(hits / total, top3_hits / total, total)
