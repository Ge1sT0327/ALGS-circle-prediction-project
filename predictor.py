"""Rule-based baseline predictor and model registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from data_models import CircleState, DEFAULT_ZONES


# ---------------------------------------------------------------------------
# simple rule-based baseline
# ---------------------------------------------------------------------------

class SimpleCirclePredictor:
    """Lightweight baseline using heuristics: position, radius, stage, map priors."""

    def __init__(self) -> None:
        self.map_bias: Dict[str, Dict[str, float]] = {
            "worlds_edge": {"north": 0.18, "south": 0.17, "east": 0.22, "west": 0.21, "center": 0.22},
            "storm_point": {"north": 0.20, "south": 0.20, "east": 0.20, "west": 0.18, "center": 0.22},
            "broken_moon": {"north": 0.19, "south": 0.19, "east": 0.20, "west": 0.20, "center": 0.22},
            "e_district":  {"north": 0.20, "south": 0.20, "east": 0.20, "west": 0.20, "center": 0.20},
        }

    def _base_scores(self, map_name: str) -> Dict[str, float]:
        return dict(self.map_bias.get(map_name, {z: 0.2 for z in DEFAULT_ZONES}))

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        clean = {z: max(s, 0.0) for z, s in scores.items()}
        total = sum(clean.values()) or 1.0
        return {z: s / total for z, s in clean.items()}

    def predict(self, state: CircleState) -> Tuple[str, Dict[str, float]]:
        scores = self._base_scores(state.map_name)

        if state.circle_y > 0:
            scores["south"] += 0.08; scores["north"] -= 0.03
        else:
            scores["north"] += 0.08; scores["south"] -= 0.03

        if state.circle_x > 0:
            scores["west"] += 0.05; scores["east"] -= 0.02
        else:
            scores["east"] += 0.05; scores["west"] -= 0.02

        if state.circle_radius > 2800:
            scores["center"] += 0.08
        elif state.circle_radius < 1800:
            scores["center"] -= 0.03
            scores["north"] += 0.02; scores["south"] += 0.02

        if state.stage >= 4:
            scores["center"] += 0.05

        probs = self._normalize(scores)
        return max(probs, key=probs.get), probs


def make_default_predictor() -> SimpleCirclePredictor:
    return SimpleCirclePredictor()
