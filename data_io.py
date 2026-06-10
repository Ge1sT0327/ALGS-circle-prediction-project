from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List

from data_models import CircleRecord


CSV_HEADERS = [
    "match_id",
    "map_name",
    "stage",
    "circle_x",
    "circle_y",
    "circle_radius",
    "next_circle_x",
    "next_circle_y",
    "next_circle_radius",
    "final_zone",
]


def save_records_csv(records: Iterable[CircleRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "match_id": record.match_id,
                    "map_name": record.map_name,
                    "stage": record.stage,
                    "circle_x": record.circle_x,
                    "circle_y": record.circle_y,
                    "circle_radius": record.circle_radius,
                    "next_circle_x": record.next_circle_x,
                    "next_circle_y": record.next_circle_y,
                    "next_circle_radius": record.next_circle_radius,
                    "final_zone": record.final_zone,
                }
            )


def load_records_csv(path: str | Path) -> List[CircleRecord]:
    path = Path(path)
    records: List[CircleRecord] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                CircleRecord(
                    match_id=row["match_id"],
                    map_name=row["map_name"],
                    stage=int(row["stage"]),
                    circle_x=float(row["circle_x"]),
                    circle_y=float(row["circle_y"]),
                    circle_radius=float(row["circle_radius"]),
                    next_circle_x=float(row["next_circle_x"]),
                    next_circle_y=float(row["next_circle_y"]),
                    next_circle_radius=float(row["next_circle_radius"]),
                    final_zone=row["final_zone"],
                )
            )
    return records
