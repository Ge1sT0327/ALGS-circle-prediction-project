"""
Vision-based ring predictor using DINOv2 for terrain-aware heatmap generation.

Input:  map name + current ring (x, y, r)
Output: heatmap predictions for all subsequent rings, terrain-aware.

DINOv2 (self-supervised ViT) understands terrain without labels:
  - Water bodies, mountains, buildings all have distinct feature signatures
  - The model implicitly learns playable vs unplayable regions
  - Combined with ring-pull physics to generate realistic predictions
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from curl_cffi import requests as cffi_requests

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COORD_SPACE = 16384

# Apex ring shrink ratios (approximate, per-game mechanics)
RING_SHRINK = [0.50, 0.60, 0.60, 0.50]  # R1->R2, R2->R3, R3->R4, R4->R5

MAP_URLS = {
    "worlds_edge": "mp_rr_desertlands_hu.png",
    "storm_point": "mp_rr_tropic_island_mu2.png",
    "e_district":  "mp_rr_district.png",
}

# ---------------------------------------------------------------------------
# DINOv2 terrain feature extractor
# ---------------------------------------------------------------------------

class TerrainAnalyzer:
    """Use DINOv2 to analyze terrain from map images — zero-shot, no labels."""

    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.model = None
        self.patch_size = 14
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        logger.info("Loading DINOv2-%s via transformers...", self.model_size)
        from transformers import AutoImageProcessor, AutoModel
        model_id = f"facebook/dinov2-{self.model_size}"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(DEVICE).eval()
        self._loaded = True

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extract dense patch features (H_patches x W_patches x D)."""
        self._ensure_loaded()

        # Resize to multiple of patch_size
        W, H = image.size
        new_W = (W // self.patch_size) * self.patch_size
        new_H = (H // self.patch_size) * self.patch_size
        img_resized = image.resize((new_W, new_H))

        inputs = self.processor(images=img_resized, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        patches = outputs.last_hidden_state[:, 1:, :]  # (1, N_patches, D)
        N, D = patches.shape[1], patches.shape[2]

        h_patches = new_H // self.patch_size
        w_patches = new_W // self.patch_size

        if h_patches * w_patches != N:
            # Fallback: infer from square root
            h_patches = w_patches = int(np.sqrt(N))
            if h_patches * w_patches != N:
                logger.warning("Patch count mismatch: %d vs %dx%d, adjusting image", N, h_patches, w_patches)
                h_patches = int(np.sqrt(N * new_H / new_W))
                w_patches = N // h_patches

        feature_map = patches.reshape(1, h_patches, w_patches, D)
        return feature_map[0].cpu().numpy()

    @torch.no_grad()
    def terrain_score(self, image: Image.Image) -> np.ndarray:
        """Compute a 'playability' heatmap (0=unplayable, 1=playable).

        Uses DINOv2 feature variance as a proxy for terrain complexity:
          - Water (flat, uniform) → low variance → unplayable
          - Mountains/rocks (textured) → high variance → playable
          - Buildings (geometric edges) → medium-high variance → playable
        """
        features = self.extract_features(image)  # (Hp, Wp, D)
        # Local feature variance as terrain complexity proxy
        from scipy.ndimage import uniform_filter
        sq_mean = uniform_filter(features.astype(np.float64), size=3, axes=(0, 1))
        mean_sq = uniform_filter(features.astype(np.float64) ** 2, size=3, axes=(0, 1))
        variance = np.mean(mean_sq - sq_mean ** 2, axis=-1)

        # Normalize to [0, 1]
        vmin, vmax = np.percentile(variance, 2), np.percentile(variance, 98)
        score = np.clip((variance - vmin) / max(vmax - vmin, 1e-8), 0, 1)

        # Upsample back to original image size
        from scipy.ndimage import zoom
        H, W = image.size[1], image.size[0]
        zoom_h = H / score.shape[0]
        zoom_w = W / score.shape[1]
        score_full = zoom(score, (zoom_h, zoom_w), order=1)
        return score_full


# ---------------------------------------------------------------------------
# ring-pull simulator with terrain constraints
# ---------------------------------------------------------------------------

class RingPullSimulator:
    """Constrained Monte Carlo ring-pull sampling.

    Given a current ring and terrain data, samples plausible next-ring positions
    following Apex game mechanics:
      - Next ring must be inside current ring (with margin)
      - Next ring center must be within current ring
      - Final ring area must have sufficient playable terrain
    """

    def __init__(self, terrain_analyzer: TerrainAnalyzer, n_samples: int = 2000):
        self.terrain = terrain_analyzer
        self.n_samples = n_samples
        self._map_cache: Dict[str, np.ndarray] = {}

    def _load_map_terrain(self, map_name: str) -> np.ndarray:
        if map_name in self._map_cache:
            return self._map_cache[map_name]

        url = MAP_URLS.get(map_name.lower())
        if url is None:
            raise ValueError(f"Unknown map: {map_name}")

        path = Path("map_images") / url
        if not path.exists():
            logger.info("Downloading %s...", url)
            resp = cffi_requests.get(
                f"https://apexlegendsstatus.com/dgs/{url}",
                impersonate="chrome124", timeout=120,
            )
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(resp.content)

        img = Image.open(path).convert("RGB")
        terrain = self.terrain.terrain_score(img)
        self._map_cache[map_name] = terrain
        return terrain

    def predict_next_rings(
        self,
        map_name: str,
        current_ring: dict,  # {"x": float, "y": float, "r": float}
        current_stage: int = 1,  # 1-indexed: 1=biggest ring, 5=smallest
    ) -> List[np.ndarray]:
        """Generate heatmaps for all remaining rings.

        Returns:
            List of heatmaps, one per remaining ring (including current if stage=1).
            Each heatmap is (H, W) probability density.
        """
        terrain = self._load_map_terrain(map_name)
        H, W = terrain.shape

        remaining_stages = list(range(current_stage, 5))  # rings 1-5 are white
        shrink_indices = range(current_stage - 1, 4)

        heatmaps = []
        cx, cy, cr = current_ring["x"], current_ring["y"], current_ring["r"]

        for stage_idx in range(len(remaining_stages)):
            shrink = RING_SHRINK[shrink_indices[stage_idx]]
            next_r = cr * shrink

            # Sample many candidate next-ring centers
            heatmap = self._sample_ring_positions(
                terrain, W, H, cx, cy, cr, next_r
            )
            heatmaps.append(heatmap)

            # Pick the most likely center for the next iteration
            cy_arr, cx_arr = np.unravel_index(heatmap.argmax(), heatmap.shape)
            cx = cx_arr / W * COORD_SPACE
            cy = cy_arr / H * COORD_SPACE
            cr = next_r

        return heatmaps

    def _sample_ring_positions(
        self,
        terrain: np.ndarray,
        W: int, H: int,
        cx: float, cy: float, cr: float,
        next_r: float,
    ) -> np.ndarray:
        """Monte Carlo sample next-ring positions, weighted by terrain playability."""
        # Convert to pixel coords
        cx_px = int(cx / COORD_SPACE * W)
        cy_px = int(cy / COORD_SPACE * H)
        cr_px = int(cr / COORD_SPACE * W)
        nr_px = int(next_r / COORD_SPACE * W)

        heatmap = np.zeros((H, W), dtype=np.float64)

        # Sample candidate centers within the current ring
        # The next ring must be fully inside the current ring
        max_center_dist = cr_px - nr_px
        if max_center_dist < 1:
            max_center_dist = cr_px // 2

        # Generate candidates
        n_attempts = 0
        n_accepted = 0
        rng = np.random.RandomState(42)

        while n_accepted < self.n_samples and n_attempts < self.n_samples * 5:
            n_attempts += 1

            # Sample a candidate center
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0, max_center_dist)
            cand_x = int(cx_px + dist * np.cos(angle))
            cand_y = int(cy_px + dist * np.sin(angle))

            # Check bounds
            if cand_x < 0 or cand_x >= W or cand_y < 0 or cand_y >= H:
                continue

            # Check that next ring is fully inside current ring
            # (simplified: center must be within max_center_dist)
            ring_dist = np.sqrt((cand_x - cx_px) ** 2 + (cand_y - cy_px) ** 2)
            if ring_dist > max_center_dist:
                continue

            # Score by terrain playability within the proposed ring
            y0, y1 = max(0, cand_y - nr_px), min(H, cand_y + nr_px)
            x0, x1 = max(0, cand_x - nr_px), min(W, cand_x + nr_px)
            if y1 <= y0 or x1 <= x0:
                continue

            # Create circular mask for the ring area
            yy, xx = np.ogrid[:y1 - y0, :x1 - x0]
            ring_mask = ((yy - nr_px) ** 2 + (xx - nr_px) ** 2) <= nr_px ** 2

            if not ring_mask.any():
                continue

            terrain_patch = terrain[y0:y1, x0:x1]
            if ring_mask.shape != terrain_patch.shape:
                # Pad/crop to match
                min_h = min(ring_mask.shape[0], terrain_patch.shape[0])
                min_w = min(ring_mask.shape[1], terrain_patch.shape[1])
                ring_mask = ring_mask[:min_h, :min_w]
                terrain_patch = terrain_patch[:min_h, :min_w]

            playable_score = terrain_patch[ring_mask].mean()

            # Heavily penalize water (low terrain complexity)
            # Require at least some playable area
            if playable_score < 0.15:
                continue

            n_accepted += 1

            # Add to heatmap (Gaussian kernel centered at candidate)
            sigma = max(2, nr_px // 20)

            # Efficient: add to a small region around candidate
            gy0 = max(0, cand_y - sigma * 3)
            gy1 = min(H, cand_y + sigma * 3 + 1)
            gx0 = max(0, cand_x - sigma * 3)
            gx1 = min(W, cand_x + sigma * 3 + 1)

            gyy, gxx = np.ogrid[:gy1 - gy0, :gx1 - gx0]
            gauss = np.exp(-((gyy - (cand_y - gy0)) ** 2 + (gxx - (cand_x - gx0)) ** 2) / (2 * sigma ** 2))
            heatmap[gy0:gy1, gx0:gx1] += gauss * playable_score

        if heatmap.sum() > 0:
            heatmap /= heatmap.sum()

        # Smooth
        heatmap = gaussian_filter(heatmap, sigma=3.0)
        if heatmap.sum() > 0:
            heatmap /= heatmap.sum()

        return heatmap


# ---------------------------------------------------------------------------
# visualization: overlay all heatmaps on map
# ---------------------------------------------------------------------------

import torchvision.transforms as T


def render_all_ring_heatmaps(
    map_name: str,
    current_ring: dict,
    current_stage: int,
    heatmaps: List[np.ndarray],
    output_path: str | Path,
) -> Path:
    """Render all ring heatmaps as colored overlays on the base map.

    Colors:
      - Ring 2: green
      - Ring 3: yellow
      - Ring 4: orange
      - Ring 5: red
    """
    url = MAP_URLS.get(map_name.lower())
    path = Path("map_images") / url
    map_img = Image.open(path).convert("RGBA")
    W, H = map_img.size

    ring_colors = [
        (0, 255, 0, 100),    # R2: green
        (255, 255, 0, 100),  # R3: yellow
        (255, 165, 0, 100),  # R4: orange
        (255, 50, 50, 100),  # R5: red
    ]
    ring_labels = [f"Ring {current_stage + i}" for i in range(1, len(heatmaps) + 1)]

    # Draw current ring
    draw = ImageDraw.Draw(map_img)
    cx = int(current_ring["x"] / COORD_SPACE * W)
    cy = int(current_ring["y"] / COORD_SPACE * H)
    cr = int(current_ring["r"] / COORD_SPACE * W)
    draw.ellipse([cx-cr, cy-cr, cx+cr, cy+cr], outline=(255, 255, 255), width=3)
    draw.text((cx + 5, cy - 20), f"Ring {current_stage}", fill=(255, 255, 255))

    # Overlay heatmaps
    overlay = Image.new("RGBA", map_img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for i, hm in enumerate(heatmaps):
        if i >= len(ring_colors):
            break

        # Resize heatmap to image size
        from scipy.ndimage import zoom
        zoom_h = H / hm.shape[0]
        zoom_w = W / hm.shape[1]
        hm_full = zoom(hm, (zoom_h, zoom_w), order=1)
        hm_full = np.clip(hm_full / hm_full.max(), 0, 1)

        color = ring_colors[i]
        # Draw the heatmap as a semi-transparent overlay
        hm_img = Image.fromarray((hm_full * 255).astype(np.uint8), mode="L")
        colored = Image.new("RGBA", map_img.size, color)
        colored.putalpha(hm_img)
        overlay = Image.alpha_composite(overlay, colored)

        # Draw peak location
        peak_y, peak_x = np.unravel_index(hm.argmax(), hm.shape)
        peak_x = int(peak_x / hm.shape[1] * W)
        peak_y = int(peak_y / hm.shape[0] * H)
        r_size = int(RING_SHRINK[min(current_stage + i - 2, 3)] * current_ring["r"] / COORD_SPACE * W)
        overlay_draw.ellipse(
            [peak_x - r_size, peak_y - r_size, peak_x + r_size, peak_y + r_size],
            outline=color[:3], width=2,
        )
        overlay_draw.text((peak_x + 5, peak_y - 15), ring_labels[i],
                          fill=color[:3] + (255,))

    result = Image.alpha_composite(map_img, overlay)
    result = result.convert("RGB")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, "PNG")
    return output_path


# ---------------------------------------------------------------------------
# main predictor class
# ---------------------------------------------------------------------------

class VisionRingPredictor:
    """End-to-end vision-based ring predictor.

    Usage:
        predictor = VisionRingPredictor()
        heatmaps = predictor.predict("storm_point", {"x": 8000, "y": 8000, "r": 5000}, stage=1)
        predictor.render(heatmaps, "output.png")
    """

    def __init__(self, n_samples: int = 3000):
        self.terrain = TerrainAnalyzer(model_size="small")
        self.simulator = RingPullSimulator(self.terrain, n_samples=n_samples)

    def predict(
        self,
        map_name: str,
        ring: dict,    # {"x": float, "y": float, "r": float}
        stage: int = 1,
    ) -> List[np.ndarray]:
        return self.simulator.predict_next_rings(map_name, ring, stage)

    def render(
        self,
        map_name: str,
        ring: dict,
        stage: int,
        heatmaps: List[np.ndarray],
        output_path: str | Path = "prediction.png",
    ) -> Path:
        return render_all_ring_heatmaps(map_name, ring, stage, heatmaps, output_path)

    def predict_and_render(
        self,
        map_name: str,
        x: float, y: float, r: float,
        stage: int = 1,
        output_path: str | Path = "prediction.png",
    ) -> Path:
        ring = {"x": x, "y": y, "r": r}
        heatmaps = self.predict(map_name, ring, stage)
        return self.render(map_name, ring, stage, heatmaps, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Vision-based ALGS Ring Predictor")
    parser.add_argument("--map", required=True, choices=["worlds_edge", "storm_point", "e_district"])
    parser.add_argument("--x", type=float, required=True, help="Ring center X (0-16384)")
    parser.add_argument("--y", type=float, required=True, help="Ring center Y (0-16384)")
    parser.add_argument("--radius", type=float, required=True, help="Ring radius")
    parser.add_argument("--stage", type=int, default=1, help="Current ring number (1-5)")
    parser.add_argument("--output", default="prediction.png", help="Output image path")
    parser.add_argument("--samples", type=int, default=3000, help="MC samples")
    args = parser.parse_args()

    predictor = VisionRingPredictor(n_samples=args.samples)
    path = predictor.predict_and_render(
        args.map, args.x, args.y, args.radius,
        stage=args.stage, output_path=args.output,
    )
    print(f"Prediction saved to: {path}")
