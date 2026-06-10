from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from data_models import CircleRecord


RAW_DIR = Path("raw_data")
PROCESSED_DIR = Path("processed_data")


def build_dataset_manifest() -> dict:
    return {
        "project": "ALGS Circle Zone Prediction",
        "sources": [
            {
                "name": "apexlegendsstatus_map_analytics",
                "url": "https://apexlegendsstatus.com/algs/map-analytics",
                "status": "blocked_by_403",
                "notes": [
                    "Direct scraping currently fails with HTTP 403.",
                    "Use browser-captured exports, screenshots, or manually collected observations instead.",
                ],
            }
        ],
        "recommended_collection": [
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
        ],
    }


def write_manifest(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_dataset_manifest(), indent=2, ensure_ascii=False), encoding="utf-8")


def export_records_json(records: Iterable[CircleRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def import_records_json(path: str | Path) -> List[CircleRecord]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: List[CircleRecord] = []
    for item in payload:
        records.append(CircleRecord(**item))
    return records
