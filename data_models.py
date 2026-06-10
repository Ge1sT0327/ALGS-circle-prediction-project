from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


DEFAULT_ZONES: tuple[str, ...] = ("north", "south", "east", "west", "center")
DEFAULT_MAPS: tuple[str, ...] = ("worlds_edge", "storm_point", "broken_moon")


@dataclass(frozen=True)
class CircleState:
    map_name: str
    circle_x: float
    circle_y: float
    circle_radius: float
    stage: int


@dataclass(frozen=True)
class CircleRecord:
    match_id: str
    map_name: str
    stage: int
    circle_x: float
    circle_y: float
    circle_radius: float
    next_circle_x: float
    next_circle_y: float
    next_circle_radius: float
    final_zone: str


@dataclass(frozen=True)
class ZonePrediction:
    predicted_zone: str
    probabilities: Dict[str, float]
    top_zones: List[str]
