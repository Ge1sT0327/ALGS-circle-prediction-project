"""
ALGS circle/ring data scraper using the Ring Guessr API.

The apexlegendsstatus.com Ring Guessr game serves real ALGS match ring data
via a JSON API. This module extracts ring positions, radii, and game metadata
by interacting with the same endpoints the browser-based game uses.
"""

from __future__ import annotations

import json
import re
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from curl_cffi import requests as cffi_requests

from data_io import save_records_csv
from data_models import CircleRecord

logger = logging.getLogger(__name__)

BASE_URL = "https://apexlegendsstatus.com"

MAP_NAME_MAPPING: Dict[str, str] = {
    # World's Edge
    "mp_rr_desertlands_hu": "worlds_edge",
    "mp_rr_desertlands_mu1": "worlds_edge",
    "mp_rr_desertlands_mu2": "worlds_edge",
    "mp_rr_desertlands_mu3": "worlds_edge",
    # Storm Point
    "mp_rr_tropical": "storm_point",
    "mp_rr_tropical_mu1": "storm_point",
    "mp_rr_tropical_mu2": "storm_point",
    "mp_rr_tropic_island": "storm_point",
    "mp_rr_tropic_island_mu1": "storm_point",
    # Broken Moon
    "mp_rr_ashs_redemption": "broken_moon",
    "mp_rr_ashs_redemption_mu1": "broken_moon",
    "mp_rr_ashs_redemption_mu2": "broken_moon",
    # E-District
    "mp_rr_aqueduct": "e_district",
    "mp_rr_aqueduct_mu1": "e_district",
    "mp_rr_district": "e_district",
    "mp_rr_district_mu1": "e_district",
    # Kings Canyon
    "mp_rr_canyonlands": "kings_canyon",
    "mp_rr_canyonlands_mu1": "kings_canyon",
    "mp_rr_canyonlands_mu2": "kings_canyon",
    # Olympus
    "mp_rr_olympus": "olympus",
    "mp_rr_olympus_mu1": "olympus",
    "mp_rr_olympus_mu2": "olympus",
}

_map_prefix_strip = re.compile(r"^mp_rr_")
_map_suffix_strip = re.compile(r"_(?:mu\d+|hu|landscape)$")

ZONE_NAMES: List[str] = ["north", "south", "east", "west", "center"]

COLLECTED_DIR = Path("collected_data")
SEEN_FILE = COLLECTED_DIR / "seen_games.json"


def _pixel_to_zone(x: float, y: float) -> str:
    """Convert pixel coords (0–16384) to a zone label."""
    cx, cy = x / 16384.0, y / 16384.0
    if 0.35 <= cx <= 0.65 and 0.35 <= cy <= 0.65:
        return "center"
    dx, dy = cx - 0.5, cy - 0.5
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    else:
        return "south" if dy > 0 else "north"


def _pixel_to_meters(x: float, y: float) -> tuple[float, float]:
    """Convert pixel coords to approximate in-game meters (centered)."""
    return (x / 16384.0) * 2000 - 1000, (y / 16384.0) * 2000 - 1000


def _normalize_map_name(raw: str) -> str:
    base = raw.replace("_landscape", "")
    if base in MAP_NAME_MAPPING:
        return MAP_NAME_MAPPING[base]
    base = _map_suffix_strip.sub("", base)
    if base in MAP_NAME_MAPPING:
        return MAP_NAME_MAPPING[base]
    return base


def _load_seen() -> set[str]:
    if SEEN_FILE.exists():
        try:
            return set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError):
            pass
    return set()


def _save_seen(seen: set[str]) -> None:
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEEN_FILE.write_text(json.dumps(sorted(seen), indent=2), encoding="utf-8")


@dataclass
class ScraperStats:
    games_collected: int = 0
    games_skipped: int = 0
    errors: int = 0
    maps_found: Dict[str, int] = field(default_factory=dict)


class ALGSRingScraper:
    """Scrapes ALGS circle/ring data from the Ring Guessr API."""

    def __init__(self, mode: str = "ALGS", delay: float = 0.5):
        self.mode = mode
        self.delay = delay
        self.session = cffi_requests.Session()
        self.stats = ScraperStats()

    def _start_session(self) -> Optional[str]:
        try:
            r = self.session.post(
                f"{BASE_URL}/algs/ringguesser/api/startGame?mode={self.mode}",
                impersonate="chrome124",
                timeout=15,
            )
            data = r.json()
            if data.get("success"):
                return data["token"]
            logger.warning("startGame failed: %s", data)
        except Exception as exc:
            logger.error("startGame error: %s", exc)
        return None

    def _fetch_ring(self, token: str) -> Optional[dict]:
        try:
            r = self.session.get(
                f"{BASE_URL}/algs/ringguesser/api/getRandomRing",
                params={"mode": self.mode, "token": token},
                impersonate="chrome124",
                timeout=15,
            )
            data = r.json()
            if data.get("success"):
                return data
            logger.warning("getRandomRing failed: %s", data)
        except Exception as exc:
            logger.error("getRandomRing error: %s", exc)
        return None

    def _submit_guess(self, token: str, round_num: int) -> Optional[dict]:
        try:
            r = self.session.post(
                f"{BASE_URL}/algs/ringguesser/api/submitGuess",
                impersonate="chrome124",
                json={
                    "token": token,
                    "round": round_num,
                    "guessX": 8192,
                    "guessY": 8192,
                    "time": 1.0,
                    "mode": self.mode,
                },
                timeout=15,
            )
            data = r.json()
            if data.get("success"):
                return data
            logger.warning("submitGuess failed: %s", data)
        except Exception as exc:
            logger.error("submitGuess error: %s", exc)
        return None

    def collect_one_game(self, seen: set[str]) -> Optional[CircleRecord]:
        """Collect ring data for a single random ALGS game."""
        time.sleep(self.delay)

        token = self._start_session()
        if not token:
            self.stats.errors += 1
            return None

        ring_data = self._fetch_ring(token)
        if not ring_data:
            self.stats.errors += 1
            return None

        game_id = ring_data.get("gameId", "")
        if game_id in seen:
            self.stats.games_skipped += 1
            return None

        rings = ring_data.get("rings", [])
        if len(rings) < 3:
            self.stats.errors += 1
            return None

        guess_result = self._submit_guess(token, 1)
        if not guess_result:
            self.stats.errors += 1
            return None

        final_ring = guess_result.get("finalRing", {})
        game_id = guess_result.get("gameId", game_id)
        game_desc = guess_result.get("gameDesc", "")

        if game_id in seen:
            self.stats.games_skipped += 1
            return None

        raw_map = ring_data["map"]
        map_name = _normalize_map_name(raw_map)

        self.stats.maps_found[map_name] = self.stats.maps_found.get(map_name, 0) + 1

        r1 = rings[0]
        r2 = rings[1]
        r3 = rings[2]
        fr = final_ring

        record = CircleRecord(
            match_id=game_id,
            map_name=map_name,
            stage=1,
            circle_x=r1["x"],
            circle_y=r1["y"],
            circle_radius=r1["r"],
            next_circle_x=r2["x"],
            next_circle_y=r2["y"],
            next_circle_radius=r2["r"],
            final_zone=_pixel_to_zone(fr.get("x", r3["x"]), fr.get("y", r3["y"])),
        )

        seen.add(game_id)
        self.stats.games_collected += 1

        if self.stats.games_collected % 10 == 0:
            logger.info(
                "Collected %d games (skipped: %d, errors: %d, maps: %s)",
                self.stats.games_collected,
                self.stats.games_skipped,
                self.stats.errors,
                dict(self.stats.maps_found),
            )

        return record

    def collect_games(self, target: int = 200) -> List[CircleRecord]:
        """Collect ring data from multiple random ALGS games."""
        seen = _load_seen()
        records: List[CircleRecord] = []

        logger.info(
            "Starting collection: target=%d, already seen=%d", target, len(seen)
        )

        while len(records) < target:
            record = self.collect_one_game(seen)
            if record:
                records.append(record)
                _save_seen(seen)

            if self.stats.errors > target * 3:
                logger.error("Too many errors, stopping collection")
                break

        logger.info(
            "Collection complete: %d records, %d skipped, %d errors",
            len(records),
            self.stats.games_skipped,
            self.stats.errors,
        )
        return records


FULL_RINGS_FILE = COLLECTED_DIR / "full_rings.json"


def _save_full_rings(games: list[dict]) -> None:
    COLLECTED_DIR.mkdir(parents=True, exist_ok=True)
    FULL_RINGS_FILE.write_text(json.dumps(games, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_full_rings() -> list[dict]:
    if FULL_RINGS_FILE.exists():
        try:
            return json.loads(FULL_RINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            pass
    return []


class ALGSGameCollector(ALGSRingScraper):
    """Extended scraper that also saves full ring sequences for map rendering."""

    def __init__(self, mode: str = "ALGS", delay: float = 0.5):
        super().__init__(mode=mode, delay=delay)
        self.full_games: list[dict] = []

    def collect_one_game(self, seen: set[str]) -> Optional[CircleRecord]:
        time.sleep(self.delay)

        token = self._start_session()
        if not token:
            self.stats.errors += 1
            return None

        ring_data = self._fetch_ring(token)
        if not ring_data:
            self.stats.errors += 1
            return None

        game_id = ring_data.get("gameId", "")
        if game_id in seen:
            self.stats.games_skipped += 1
            return None

        rings = ring_data.get("rings", [])
        if len(rings) < 3:
            self.stats.errors += 1
            return None

        guess_result = self._submit_guess(token, 1)
        if not guess_result:
            self.stats.errors += 1
            return None

        final_ring = guess_result.get("finalRing", {})
        game_id = guess_result.get("gameId", game_id)

        if game_id in seen:
            self.stats.games_skipped += 1
            return None

        raw_map = ring_data["map"]
        map_name = _normalize_map_name(raw_map)

        # Save full ring sequence for rendering
        all_rings = list(rings)
        if final_ring:
            all_rings.append(final_ring)
        self.full_games.append({
            "match_id": game_id,
            "map_name": map_name,
            "raw_map": raw_map,
            "rings": all_rings,
            "game_desc": guess_result.get("gameDesc", ""),
        })

        self.stats.maps_found[map_name] = self.stats.maps_found.get(map_name, 0) + 1

        r1 = rings[0]
        r2 = rings[1]
        r3 = rings[2]
        fr = final_ring

        record = CircleRecord(
            match_id=game_id,
            map_name=map_name,
            stage=1,
            circle_x=r1["x"],
            circle_y=r1["y"],
            circle_radius=r1["r"],
            next_circle_x=r2["x"],
            next_circle_y=r2["y"],
            next_circle_radius=r2["r"],
            final_zone=_pixel_to_zone(fr.get("x", r3["x"]), fr.get("y", r3["y"])),
        )

        seen.add(game_id)
        self.stats.games_collected += 1

        if self.stats.games_collected % 10 == 0:
            logger.info(
                "Collected %d games (skipped: %d, errors: %d, maps: %s)",
                self.stats.games_collected,
                self.stats.games_skipped,
                self.stats.errors,
                dict(self.stats.maps_found),
            )

        return record

    def collect_games(self, target: int = 200) -> List[CircleRecord]:
        self.full_games = _load_full_rings()
        # Add existing full_games IDs to seen set
        existing_ids = {g["match_id"] for g in self.full_games}
        seen = _load_seen() | existing_ids
        records: List[CircleRecord] = []

        logger.info(
            "Starting collection: target=%d, already seen=%d, full_games=%d",
            target, len(seen), len(self.full_games),
        )

        while len(self.full_games) < target:
            record = self.collect_one_game(seen)
            if record:
                records.append(record)
                _save_seen(seen)
            # Persist full rings incrementally
            if self.full_games and len(self.full_games) % 10 == 0:
                _save_full_rings(self.full_games)

            if self.stats.errors > target * 3:
                logger.error("Too many errors, stopping collection")
                break

        _save_full_rings(self.full_games)
        logger.info(
            "Collection complete: %d records, %d skipped, %d errors, %d full games",
            len(records), self.stats.games_skipped, self.stats.errors, len(self.full_games),
        )
        return records


def scrape_to_csv(target: int = 200, output_path: str | Path = "collected_data/rings.csv") -> Path:
    """High-level helper: collect ring data and save to CSV + full rings JSON."""
    scraper = ALGSGameCollector(mode="ALGS", delay=0.3)
    records = scraper.collect_games(target)
    path = Path(output_path)
    save_records_csv(records, path)
    _save_full_rings(scraper.full_games)
    logger.info("Saved %d records to %s, %d full games to %s",
                len(records), path, len(scraper.full_games), FULL_RINGS_FILE)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    path = scrape_to_csv(target=500)
    print(f"Data saved to {path}")
    print(f"Full ring data saved to {FULL_RINGS_FILE}")
