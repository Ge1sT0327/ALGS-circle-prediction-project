"""Core data models and constants shared across the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Tuple

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DEFAULT_ZONES: tuple[str, ...] = ("north", "south", "east", "west", "center")

# Maps currently in ALGS competitive rotation (Year 4+)
COMPETITIVE_MAPS: tuple[str, ...] = (
    "worlds_edge",
    "storm_point",
    "e_district",
)

# All known Apex maps (for data parsing compatibility)
ALL_KNOWN_MAPS: tuple[str, ...] = (
    "worlds_edge",
    "storm_point",
    "e_district",
    "broken_moon",
    "kings_canyon",
    "olympus",
)

# currently used for training (will grow as data expands)
DEFAULT_MAPS = COMPETITIVE_MAPS


# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircleState:
    """A single ring state: position + radius on a specific map."""
    map_name: str
    circle_x: float
    circle_y: float
    circle_radius: float
    stage: int


@dataclass(frozen=True)
class CircleRecord:
    """One labelled training example: current ring → next ring → final zone."""
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
    """Unified prediction result for any model."""
    predicted_zone: str
    probabilities: Dict[str, float]
    top_zones: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.top_zones:
            ranked = sorted(self.probabilities.items(), key=lambda x: -x[1])
            object.__setattr__(self, "top_zones", [z for z, _ in ranked[:3]])


# ---------------------------------------------------------------------------
# unified predictor protocol (structural duck-typing, not enforced at runtime)
# ---------------------------------------------------------------------------

class Predictor(Protocol):
    """Any model that can predict the next zone from a CircleState."""
    def predict(self, state: CircleState) -> Tuple[str, Dict[str, float]]: ...
