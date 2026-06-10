from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List

from data_models import CircleRecord, CircleState, DEFAULT_MAPS, DEFAULT_ZONES


MAP_INDEX = {name: index for index, name in enumerate(DEFAULT_MAPS)}
ZONE_INDEX = {name: index for index, name in enumerate(DEFAULT_ZONES)}


def map_to_one_hot(map_name: str) -> List[float]:
    vec = [0.0] * len(DEFAULT_MAPS)
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
    if 0 <= index < len(DEFAULT_ZONES):
        return DEFAULT_ZONES[index]
    return "unknown"
