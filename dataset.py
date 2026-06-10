from __future__ import annotations

from data_models import CircleRecord


def build_sample_dataset() -> list[CircleRecord]:
    return [
        CircleRecord("match_001", "worlds_edge", 2, 500, -800, 3200, 900, -400, 2600, "east"),
        CircleRecord("match_002", "storm_point", 3, -1200, 300, 2600, -700, 50, 2100, "north"),
        CircleRecord("match_003", "broken_moon", 5, 200, 1400, 1700, 100, 900, 1300, "center"),
        CircleRecord("match_004", "worlds_edge", 4, -300, -1500, 2400, -150, -900, 1900, "south"),
        CircleRecord("match_005", "storm_point", 6, 1800, 200, 1300, 1200, 100, 900, "west"),
    ]
