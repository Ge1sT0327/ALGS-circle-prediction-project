"""
ALGS circle/zone data sources and manifest.

Primary data source: Ring Guessr API on apexlegendsstatus.com
  - POST /algs/ringguesser/api/startGame?mode=ALGS  → session token
  - GET  /algs/ringguesser/api/getRandomRing         → ring positions (rounds 1-3)
  - POST /algs/ringguesser/api/submitGuess           → final ring + gameId

The scraper module (scraper.py) handles the full collection pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from data_models import CircleRecord

RAW_DIR = Path("raw_data")
PROCESSED_DIR = Path("processed_data")
COLLECTED_DIR = Path("collected_data")


def build_dataset_manifest() -> dict:
    return {
        "project": "ALGS Circle Zone Prediction",
        "data_sources": [
            {
                "name": "apexlegendsstatus_ring_guessr_api",
                "description": "Ring Guessr API serving real ALGS match ring data",
                "endpoints": {
                    "start_game": "POST /algs/ringguesser/api/startGame?mode=ALGS",
                    "get_ring": "GET /algs/ringguesser/api/getRandomRing",
                    "submit_guess": "POST /algs/ringguesser/api/submitGuess",
                },
                "status": "active",
                "notes": [
                    "Uses curl_cffi with TLS fingerprint impersonation (chrome124).",
                    "Ring coords in 16384x16384 pixel space; zone labels derived from pixel position.",
                    "Supports both ALGS and public game modes.",
                ],
            },
            {
                "name": "apexlegendsstatus_map_analytics",
                "url": "https://apexlegendsstatus.com/algs/map-analytics",
                "status": "blocked_by_cloudflare",
                "notes": [
                    "Direct scraping blocked by Cloudflare challenge (HTTP 403).",
                    "Ring Guessr API (above) provides equivalent data and is the recommended source.",
                ],
            },
        ],
        "collected_data_schema": [
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
    path.write_text(
        json.dumps(build_dataset_manifest(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def export_records_json(records: Iterable[CircleRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def import_records_json(path: str | Path) -> List[CircleRecord]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: List[CircleRecord] = []
    for item in payload:
        records.append(CircleRecord(**item))
    return records
