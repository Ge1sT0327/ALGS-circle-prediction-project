"""
Unsupervised / non-parametric models for ALGS ring prediction.

No zone labels required — these models learn directly from ring position data.

All predictors conform to the project convention:
    predict(state: CircleState) -> (best_zone, {zone: probability})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_models import CircleRecord, CircleState, DEFAULT_ZONES
from feature_engineering import rebuild_indices


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_features(records: List[CircleRecord]) -> np.ndarray:
    return np.array([[r.circle_x, r.circle_y, r.circle_radius] for r in records], dtype=float)


def _records_by_map(records: List[CircleRecord]) -> Dict[str, List[CircleRecord]]:
    by_map: Dict[str, List[CircleRecord]] = {}
    for r in records:
        by_map.setdefault(r.map_name, []).append(r)
    return by_map


ZONE_CENTERS_PX = {
    "north": (8192, 2048),
    "south": (8192, 14336),
    "east":  (14336, 8192),
    "west":  (2048, 8192),
    "center": (8192, 8192),
}


def _scores_to_prediction(scores: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    total = sum(scores.values()) or 1.0
    probs = {z: s / total for z, s in scores.items()}
    return max(probs, key=probs.get), probs


# ---------------------------------------------------------------------------
# GMM predictor
# ---------------------------------------------------------------------------

@dataclass
class GMMPredictor:
    """Gaussian Mixture Model — each component is a ring-location archetype."""

    map_gmms: Dict[str, GaussianMixture] = field(default_factory=dict)
    map_scalers: Dict[str, StandardScaler] = field(default_factory=dict)

    def predict(self, state: CircleState) -> Tuple[str, Dict[str, float]]:
        gmm = self.map_gmms.get(state.map_name)
        scaler = self.map_scalers.get(state.map_name)
        if gmm is None:
            return "center", {z: 0.2 for z in DEFAULT_ZONES}

        scores = {}
        for zone, (zx, zy) in ZONE_CENTERS_PX.items():
            feats = [[state.circle_x, state.circle_y, state.circle_radius],
                     [zx, zy, 3000]]
            feats_s = scaler.transform(feats)
            scores[zone] = float(gmm.score_samples(feats_s)[1])

        smax = np.exp(list(scores.values()) - np.max(list(scores.values())))
        smax /= smax.sum()
        for i, zone in enumerate(scores):
            scores[zone] = float(smax[i])

        return _scores_to_prediction(scores)


def train_gmm(records: List[CircleRecord], n_components: int = 5) -> GMMPredictor:
    predictor = GMMPredictor()
    by_map = _records_by_map(records)
    rebuild_indices(sorted(by_map.keys()), list(DEFAULT_ZONES))

    for map_name, recs in by_map.items():
        X = _build_features(recs)
        scaler = StandardScaler().fit(X)
        X_s = scaler.transform(X)
        gmm = GaussianMixture(
            n_components=min(n_components, len(recs)),
            random_state=42, covariance_type="full",
        ).fit(X_s)
        predictor.map_gmms[map_name] = gmm
        predictor.map_scalers[map_name] = scaler
    return predictor


# ---------------------------------------------------------------------------
# KDE predictor
# ---------------------------------------------------------------------------

@dataclass
class KDEPredictor:
    """Kernel Density Estimation — non-parametric ring-position density."""

    map_kdes: Dict[str, KernelDensity] = field(default_factory=dict)
    map_scalers: Dict[str, StandardScaler] = field(default_factory=dict)

    def predict(self, state: CircleState) -> Tuple[str, Dict[str, float]]:
        kde = self.map_kdes.get(state.map_name)
        scaler = self.map_scalers.get(state.map_name)
        if kde is None:
            return "center", {z: 0.2 for z in DEFAULT_ZONES}

        scores = {}
        for zone, (zx, zy) in ZONE_CENTERS_PX.items():
            feats = scaler.transform([[zx, zy, 3000]])
            scores[zone] = float(np.exp(kde.score_samples(feats)[0]))

        return _scores_to_prediction(scores)


def train_kde(records: List[CircleRecord], bandwidth: float = 0.4) -> KDEPredictor:
    predictor = KDEPredictor()
    by_map = _records_by_map(records)
    rebuild_indices(sorted(by_map.keys()), list(DEFAULT_ZONES))

    for map_name, recs in by_map.items():
        X = _build_features(recs)
        scaler = StandardScaler().fit(X)
        X_s = scaler.transform(X)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X_s)
        predictor.map_kdes[map_name] = kde
        predictor.map_scalers[map_name] = scaler
    return predictor


# ---------------------------------------------------------------------------
# K-Means archetype analysis (exploratory)
# ---------------------------------------------------------------------------

@dataclass
class KMeansArchetypes:
    map_models: Dict[str, KMeans] = field(default_factory=dict)
    map_scalers: Dict[str, StandardScaler] = field(default_factory=dict)
    cluster_labels: Dict[str, List[int]] = field(default_factory=dict)

    def describe(self, map_name: str) -> List[Dict]:
        model = self.map_models.get(map_name)
        scaler = self.map_scalers.get(map_name)
        if model is None:
            return []

        centroids = scaler.inverse_transform(model.cluster_centers_)
        labels = self.cluster_labels.get(map_name, [])
        descriptions = []
        for i, c in enumerate(centroids):
            dx, dy = c[0], c[1]
            dist = np.sqrt(dx**2 + dy**2)
            angle = np.degrees(np.arctan2(dy, dx))
            direction = (
                "N" if -45 <= angle < 45 else
                "E" if 45 <= angle < 135 else
                "S" if angle >= 135 or angle < -135 else "W"
            )
            descriptions.append({
                "cluster": i,
                "count": labels.count(i) if labels else 0,
                "dx_pixels": round(float(dx)),
                "dy_pixels": round(float(dy)),
                "distance_pixels": round(float(dist)),
                "direction": direction,
            })
        return descriptions


def train_kmeans(records: List[CircleRecord], n_clusters: int = 5) -> KMeansArchetypes:
    result = KMeansArchetypes()
    by_map = _records_by_map(records)

    for map_name, recs in by_map.items():
        X = np.array([[r.next_circle_x - r.circle_x, r.next_circle_y - r.circle_y]
                       for r in recs], dtype=float)
        scaler = StandardScaler().fit(X)
        X_s = scaler.transform(X)
        km = KMeans(n_clusters=min(n_clusters, len(recs)), random_state=42, n_init=10).fit(X_s)
        result.map_models[map_name] = km
        result.map_scalers[map_name] = scaler
        result.cluster_labels[map_name] = list(km.labels_)
    return result
