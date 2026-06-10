from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


DEFAULT_ZONES = ("north", "south", "east", "west", "center")


@dataclass(frozen=True)
class CircleState:
    map_name: str
    circle_x: float
    circle_y: float
    circle_radius: float
    stage: int


class SimpleCirclePredictor:
    """A lightweight baseline predictor for ALGS circle zones.

    The goal of this starter project is not to be highly accurate.
    Instead, it provides a small, understandable baseline that can be
    extended later with real match data and machine learning models.
    """

    def __init__(self) -> None:
        self.map_bias: Dict[str, Dict[str, float]] = {
            "worlds_edge": {
                "north": 0.18,
                "south": 0.17,
                "east": 0.22,
                "west": 0.21,
                "center": 0.22,
            },
            "storm_point": {
                "north": 0.20,
                "south": 0.20,
                "east": 0.20,
                "west": 0.18,
                "center": 0.22,
            },
            "broken_moon": {
                "north": 0.19,
                "south": 0.19,
                "east": 0.20,
                "west": 0.20,
                "center": 0.22,
            },
        }

    def _base_scores(self, map_name: str) -> Dict[str, float]:
        return dict(self.map_bias.get(map_name, {zone: 0.2 for zone in DEFAULT_ZONES}))

    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        clean_scores = {zone: max(value, 0.0) for zone, value in scores.items()}
        total = sum(clean_scores.values()) or 1.0
        return {zone: value / total for zone, value in clean_scores.items()}

    def predict_zone(self, state: CircleState) -> Tuple[str, Dict[str, float]]:
        scores = self._base_scores(state.map_name)

        # Position hints: if the current circle is on one side of the map,
        # the next circle often shifts in the opposite direction.
        if state.circle_y > 0:
            scores["south"] += 0.08
            scores["north"] -= 0.03
        else:
            scores["north"] += 0.08
            scores["south"] -= 0.03

        if state.circle_x > 0:
            scores["west"] += 0.05
            scores["east"] -= 0.02
        else:
            scores["east"] += 0.05
            scores["west"] -= 0.02

        # Circle size hints.
        if state.circle_radius > 2800:
            scores["center"] += 0.08
        elif state.circle_radius < 1800:
            scores["center"] -= 0.03
            scores["north"] += 0.02
            scores["south"] += 0.02

        # Late stages tend to tighten toward more central zones.
        if state.stage >= 4:
            scores["center"] += 0.05

        normalized = self._normalize(scores)
        best_zone = max(normalized, key=normalized.get)
        return best_zone, normalized


def build_demo_states() -> List[CircleState]:
    return [
        CircleState("worlds_edge", 500, -800, 3200, 2),
        CircleState("storm_point", -1200, 300, 2600, 3),
        CircleState("broken_moon", 200, 1400, 1700, 5),
    ]
