"""Feature extraction for ML models."""

from __future__ import annotations

from typing import List

from data_models import CircleRecord, CircleState, DEFAULT_MAPS, DEFAULT_ZONES

# Maps and zones are indexed dynamically from training data
MAP_INDEX = {name: i for i, name in enumerate(DEFAULT_MAPS)}
ZONE_INDEX = {name: i for i, name in enumerate(DEFAULT_ZONES)}


def rebuild_indices(maps: list[str], zones: list[str] | None = None) -> None:
    """Rebuild MAP_INDEX / ZONE_INDEX from data (call after loading dataset)."""
    global MAP_INDEX, ZONE_INDEX
    MAP_INDEX = {name: i for i, name in enumerate(maps)}
    if zones:
        ZONE_INDEX = {name: i for i, name in enumerate(zones)}


def map_to_one_hot(map_name: str) -> List[float]:
    vec = [0.0] * len(MAP_INDEX)
    if map_name in MAP_INDEX:
        vec[MAP_INDEX[map_name]] = 1.0
    return vec


def stage_bucket(stage: int) -> List[float]:
    if stage <= 2:
        return [1.0, 0.0, 0.0]
    if stage <= 4:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def extract_state_features(state: CircleState) -> List[float]:
    return [
        state.circle_x / 10000.0,
        state.circle_y / 10000.0,
        state.circle_radius / 10000.0,
        state.stage / 10.0,
        *map_to_one_hot(state.map_name),
        *stage_bucket(state.stage),
    ]


def extract_record_features(record: CircleRecord) -> List[float]:
    state = CircleState(
        map_name=record.map_name,
        circle_x=record.circle_x,
        circle_y=record.circle_y,
        circle_radius=record.circle_radius,
        stage=record.stage,
    )
    return extract_state_features(state)


def label_to_index(label: str) -> int:
    return ZONE_INDEX.get(label, len(DEFAULT_ZONES))


def index_to_label(index: int) -> str:
    zones = list(ZONE_INDEX.keys())
    if 0 <= index < len(zones):
        return zones[index]
    return "unknown"
