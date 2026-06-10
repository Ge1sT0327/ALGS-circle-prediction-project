"""
Unsupervised / non-parametric models for ALGS ring prediction.

No zone labels needed — these models learn directly from ring position data:
  - Kernel Density Estimation (KDE): probability density of next-ring positions
  - Gaussian Mixture Model (GMM): soft clustering of pull patterns
  - K-Means: discover natural ring-pull archetypes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_models import CircleRecord, CircleState, ZonePrediction
from data_io import load_records_csv, save_records_csv

__all__ = [
    "KDERingPredictor",
    "GMMRingPredictor",
    "KMeansRingArchetypes",
    "train_kde_predictor",
    "train_gmm_predictor",
    "train_kmeans_archetypes",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_pull_vectors(records: List[CircleRecord]) -> np.ndarray:
    """Extract (dx, dy) pull vectors from records: next_circle - current_circle."""
    vecs = []
    for r in records:
        vecs.append([r.next_circle_x - r.circle_x, r.next_circle_y - r.circle_y])
    return np.array(vecs, dtype=float)


def _build_absolute_positions(records: List[CircleRecord]) -> np.ndarray:
    """Extract next-circle absolute (x, y) positions."""
    return np.array([[r.next_circle_x, r.next_circle_y] for r in records], dtype=float)


def _build_features(records: List[CircleRecord]) -> np.ndarray:
    """Full feature vector: [circle_x, circle_y, circle_radius]."""
    return np.array(
        [[r.circle_x, r.circle_y, r.circle_radius] for r in records],
        dtype=float,
    )


def _records_by_map(records: List[CircleRecord]) -> Dict[str, List[CircleRecord]]:
    by_map: Dict[str, List[CircleRecord]] = {}
    for r in records:
        by_map.setdefault(r.map_name, []).append(r)
    return by_map


def _top_zones(scores: Dict[str, float], k: int = 3) -> List[str]:
    return [z for z, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]


# ---------------------------------------------------------------------------
# KDE-based predictor
# ---------------------------------------------------------------------------

@dataclass
class KDERingPredictor:
    """Predict next ring location via Kernel Density Estimation.

    For a given current-ring state, fits a KDE over the *pull vectors*
    (dx, dy) observed in the training data for that map. The highest-density
    region in KDE space corresponds to the most likely pull direction/distance.
    """

    map_kdes: Dict[str, KernelDensity] = field(default_factory=dict)
    map_scalers: Dict[str, StandardScaler] = field(default_factory=dict)
    default_zones: Tuple[str, ...] = ("north", "south", "east", "west", "center")

    def predict_position(self, state: CircleState) -> Tuple[float, float, float]:
        """Return predicted (next_x, next_y, confidence) for a CircleState."""
        kde = self.map_kdes.get(state.map_name)
        scaler = self.map_scalers.get(state.map_name)
        if kde is None:
            return state.circle_x, state.circle_y, 0.0

        feat = scaler.transform([[state.circle_x, state.circle_y, state.circle_radius]])
        log_dens = kde.score_samples(feat)[0]
        return state.circle_x, state.circle_y, float(np.exp(log_dens))

    def predict_zone(self, state: CircleState, records: List[CircleRecord]) -> ZonePrediction:
        """Use KDE density over pull targets to score 5 coarse zones."""
        map_recs = [r for r in records if r.map_name == state.map_name]
        if not map_recs:
            return ZonePrediction("center", {z: 0.2 for z in self.default_zones}, list(self.default_zones))

        kde = self.map_kdes.get(state.map_name)
        scaler = self.map_scalers.get(state.map_name)
        if kde is None:
            return ZonePrediction("center", {z: 0.2 for z in self.default_zones}, list(self.default_zones))

        scores = {}
        zone_centers = {
            "north": (8192, 2048),
            "south": (8192, 14336),
            "east": (14336, 8192),
            "west": (2048, 8192),
            "center": (8192, 8192),
        }
        for zone, (zx, zy) in zone_centers.items():
            feat = scaler.transform([[zx, zy, 3000]])
            scores[zone] = float(np.exp(kde.score_samples(feat)[0]))

        total = sum(scores.values()) or 1.0
        probs = {z: s / total for z, s in scores.items()}
        best = max(probs, key=probs.get)
        return ZonePrediction(best, probs, _top_zones(probs))


def train_kde_predictor(records: List[CircleRecord], bandwidth: float = 0.3) -> KDERingPredictor:
    """Train per-map KDE models on pull vectors."""
    predictor = KDERingPredictor()
    by_map = _records_by_map(records)

    for map_name, recs in by_map.items():
        X = _build_features(recs)
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X_scaled)
        predictor.map_kdes[map_name] = kde
        predictor.map_scalers[map_name] = scaler

    return predictor


# ---------------------------------------------------------------------------
# GMM-based predictor
# ---------------------------------------------------------------------------

@dataclass
class GMMRingPredictor:
    """Probabilistic ring-pull predictor using Gaussian Mixture Models.

    Each GMM component represents a "pull archetype" — a common pattern
    of where rings move next from a given configuration. The model gives
    soft probabilities over components.
    """

    map_gmms: Dict[str, GaussianMixture] = field(default_factory=dict)
    map_scalers: Dict[str, StandardScaler] = field(default_factory=dict)
    n_components: int = 5

    def predict_zone(self, state: CircleState) -> ZonePrediction:
        """Score zones by component responsibility."""
        gmm = self.map_gmms.get(state.map_name)
        scaler = self.map_scalers.get(state.map_name)
        zones = ("north", "south", "east", "west", "center")

        if gmm is None:
            return ZonePrediction("center", {z: 0.2 for z in zones}, list(zones))

        feat = scaler.transform([[state.circle_x, state.circle_y, state.circle_radius]])
        resp = gmm.predict_proba(feat)[0]

        zone_centroids = {
            "north": (8192, 2048),
            "south": (8192, 14336),
            "east": (14336, 8192),
            "west": (2048, 8192),
            "center": (8192, 8192),
        }
        component_centers = gmm.means_
        component_to_zone = {}
        for i, center in enumerate(component_centers):
            cx, cy = center[0], center[1]
            best_zone = min(
                zone_centroids, key=lambda z: (cx - zone_centroids[z][0]) ** 2 + (cy - zone_centroids[z][1]) ** 2
            )
            component_to_zone[i] = best_zone

        scores = {z: 0.0 for z in zones}
        for i, r_val in enumerate(resp):
            scores[component_to_zone[i]] += r_val

        total = sum(scores.values()) or 1.0
        probs = {z: s / total for z, s in scores.items()}
        best = max(probs, key=probs.get)
        return ZonePrediction(best, probs, _top_zones(probs))


def train_gmm_predictor(
    records: List[CircleRecord], n_components: int = 5, random_state: int = 42
) -> GMMRingPredictor:
    """Train per-map GMM models on ring features."""
    predictor = GMMRingPredictor(n_components=n_components)
    by_map = _records_by_map(records)

    for map_name, recs in by_map.items():
        X = _build_features(recs)
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        gmm = GaussianMixture(
            n_components=min(n_components, len(recs)),
            random_state=random_state,
            covariance_type="full",
        ).fit(X_scaled)
        predictor.map_gmms[map_name] = gmm
        predictor.map_scalers[map_name] = scaler

    return predictor


# ---------------------------------------------------------------------------
# K-Means ring-pull archetypes (exploratory)
# ---------------------------------------------------------------------------

@dataclass
class KMeansRingArchetypes:
    """Discover natural ring-pull archetypes via K-Means clustering.

    Each cluster centroid represents a common "pull pattern" — a typical
    (dx, dy) vector from current ring to next ring.
    """

    map_models: Dict[str, KMeans] = field(default_factory=dict)
    map_scalers: Dict[str, StandardScaler] = field(default_factory=dict)
    cluster_labels: Dict[str, List[int]] = field(default_factory=dict)

    def describe(self, map_name: str) -> List[Dict]:
        """Return human-readable descriptions of pull archetypes for a map."""
        model = self.map_models.get(map_name)
        scaler = self.map_scalers.get(map_name)
        if model is None:
            return []

        centroids_scaled = model.cluster_centers_
        # Inverse-transform centroids back to pixel space
        centroids = scaler.inverse_transform(centroids_scaled)
        labels = self.cluster_labels.get(map_name, [])
        descriptions = []
        for i, center in enumerate(centroids):
            dx, dy = center[0], center[1]
            dist = np.sqrt(dx**2 + dy**2)
            angle = np.degrees(np.arctan2(dy, dx))
            count = labels.count(i) if labels else 0
            direction = (
                "N" if -45 <= angle < 45 else
                "E" if 45 <= angle < 135 else
                "S" if angle >= 135 or angle < -135 else
                "W"
            )
            descriptions.append({
                "cluster": i,
                "count": count,
                "dx_pixels": round(float(dx)),
                "dy_pixels": round(float(dy)),
                "distance_pixels": round(float(dist)),
                "direction": direction,
            })
        return descriptions


def train_kmeans_archetypes(
    records: List[CircleRecord], n_clusters: int = 5, random_state: int = 42
) -> KMeansRingArchetypes:
    """Cluster pull vectors to discover natural ring-pull patterns."""
    result = KMeansRingArchetypes()
    by_map = _records_by_map(records)

    for map_name, recs in by_map.items():
        X = _build_pull_vectors(recs)
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        kmeans = KMeans(
            n_clusters=min(n_clusters, len(recs)),
            random_state=random_state,
            n_init=10,
        ).fit(X_scaled)

        result.map_models[map_name] = kmeans
        result.map_scalers[map_name] = scaler
        result.cluster_labels[map_name] = list(kmeans.labels_)

    return result
