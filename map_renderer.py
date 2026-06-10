"""
Download ALGS map images and render ring overlays using GPU acceleration.

Base maps are served from apexlegendsstatus.com/dgs/ as large PNG files
(~24 MB, ~8000x8000 px). Ring coordinates use a 16384x16384 pixel space.

GPU: PyTorch CUDA for compositing (alpha blending, circle rasterization).
PNG encoding remains CPU-bound via Pillow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from curl_cffi import requests as cffi_requests

logger = logging.getLogger(__name__)

BASE_URL = "https://apexlegendsstatus.com"
MAPS_DIR = Path("map_images")
RENDER_DIR = Path("rendered_games")

COORD_SPACE = 16384

MAP_FILE_MAPPING: Dict[str, str] = {
    "worlds_edge": "mp_rr_desertlands_hu.png",
    "storm_point": "mp_rr_tropic_island_mu2.png",
    "e_district": "mp_rr_district.png",
    "broken_moon": "mp_rr_ashs_redemption.png",
}

RING_COLORS_GPU = {
    0: (1.0, 1.0, 1.0, 0.31),   # round 1: white
    1: (1.0, 1.0, 1.0, 0.39),   # round 2: white
    2: (1.0, 1.0, 1.0, 0.47),   # round 3: white
    3: (1.0, 0.78, 0.20, 0.55), # round 4: orange
    4: (1.0, 0.33, 0.33, 0.71), # final: red
}

RING_OUTLINES_GPU = {
    0: (0.78, 0.78, 0.78),
    1: (0.78, 0.78, 0.78),
    2: (0.78, 0.78, 0.78),
    3: (0.78, 0.59, 0.12),
    4: (0.78, 0.20, 0.20),
}


class MapRendererGPU:
    """Download map images and render ring overlays — GPU accelerated."""

    def __init__(self, base_dir: str | Path = "."):
        self.base = Path(base_dir)
        self.maps_dir = self.base / MAPS_DIR
        self.render_dir = self.base / RENDER_DIR
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        self.render_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_cache: Dict[str, torch.Tensor] = {}
        self._cpu_cache: Dict[str, Image.Image] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Renderer using device: %s", self._device)

    def _map_file(self, map_name: str) -> str:
        return MAP_FILE_MAPPING.get(map_name.lower(), f"{map_name}.png")

    def download_map(self, map_name: str) -> Optional[Path]:
        filename = self._map_file(map_name)
        local_path = self.maps_dir / filename

        if local_path.exists() and local_path.stat().st_size > 100000:
            return local_path

        url = f"{BASE_URL}/dgs/{filename}"
        logger.info("Downloading map: %s", url)

        try:
            resp = cffi_requests.get(url, impersonate="chrome124", timeout=120)
            if resp.status_code == 200 and len(resp.content) > 100000:
                local_path.write_bytes(resp.content)
                logger.info("Saved: %s (%d bytes)", local_path, len(resp.content))
                return local_path
        except Exception as exc:
            logger.error("Map download error: %s", exc)
        return None

    def _load_map_gpu(self, map_name: str) -> Optional[torch.Tensor]:
        """Load map as GPU RGBA tensor (H, W, 4) float32 [0, 1]."""
        if map_name in self._gpu_cache:
            return self._gpu_cache[map_name]

        path = self.download_map(map_name)
        if path is None:
            return None

        img = Image.open(path).convert("RGBA")
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).to(self._device)
        self._gpu_cache[map_name] = tensor
        self._cpu_cache[map_name] = img
        return tensor

    def _draw_circles_gpu(
        self, base: torch.Tensor, rings: list[dict], img_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Draw ring circles onto the base map tensor, all on GPU.

        Returns (H, W, 3) uint8 tensor on CPU ready for PIL save.
        """
        H, W = base.shape[0], base.shape[1]
        device = base.device

        # Use base RGB as starting canvas
        result = base[:, :, :3].clone()

        # Precompute coordinate grids once
        y_grid = torch.arange(H, device=device, dtype=torch.float32)
        x_grid = torch.arange(W, device=device, dtype=torch.float32)

        for i, ring in enumerate(rings):
            cx = int(ring["x"] / COORD_SPACE * W)
            cy = int(ring["y"] / COORD_SPACE * H)
            cr = max(1, int(ring["r"] / COORD_SPACE * W))

            color_idx = min(i, 4)
            fc = RING_COLORS_GPU[color_idx]
            oc = RING_OUTLINES_GPU[color_idx]

            # Bounding box for this ring
            y0, y1 = max(0, cy - cr - 3), min(H, cy + cr + 3)
            x0, x1 = max(0, cx - cr - 3), min(W, cx + cr + 3)

            if y1 <= y0 or x1 <= x0:
                continue

            # Local coordinate grids within bounding box
            yy = y_grid[y0:y1][:, None]  # (h, 1)
            xx = x_grid[None, x0:x1]      # (1, w)

            dist2 = (yy - cy) ** 2 + (xx - cx) ** 2

            # Fill mask
            fill_mask = dist2 <= cr ** 2
            # Outline mask (ring border, 3px wide)
            outline_inner = dist2 <= (cr - 3) ** 2
            outline_mask = fill_mask & (~outline_inner)

            # Alpha for fill (inside ring, not outline)
            inner_mask = fill_mask & (~outline_mask)

            # Composite fill
            alpha_f = fc[3]
            for c in range(3):
                result[y0:y1, x0:x1, c] = torch.where(
                    inner_mask,
                    fc[c] * alpha_f + result[y0:y1, x0:x1, c] * (1 - alpha_f),
                    result[y0:y1, x0:x1, c],
                )

            # Composite outline
            for c in range(3):
                result[y0:y1, x0:x1, c] = torch.where(
                    outline_mask,
                    oc[c],
                    result[y0:y1, x0:x1, c],
                )

        # Convert to CPU uint8
        result = (result.clamp(0, 1) * 255).to(torch.uint8).cpu()
        return result

    def render_game(
        self,
        match_id: str,
        map_name: str,
        rings: list[dict],
        output_path: Optional[str | Path] = None,
    ) -> Optional[Path]:
        """Draw ring circles on the base map and save — GPU path."""
        base = self._load_map_gpu(map_name)
        if base is None:
            logger.error("Cannot load map image for %s", map_name)
            return None

        img_size = (base.shape[1], base.shape[0])
        gpu_result = self._draw_circles_gpu(base, rings, img_size)

        # Convert GPU tensor to PIL Image for text labels (CPU only)
        result_np = gpu_result.numpy()
        img = Image.fromarray(result_np)

        # Draw text labels using PIL (quick CPU step)
        draw = ImageDraw.Draw(img)
        for i, ring in enumerate(rings):
            cx = int(ring["x"] / COORD_SPACE * img_size[0])
            cy = int(ring["y"] / COORD_SPACE * img_size[1])
            cr = max(1, int(ring["r"] / COORD_SPACE * img_size[0]))

            label = f"R{i + 1}" if i < 4 else "Final"
            try:
                font = ImageFont.truetype("arial.ttf", size=max(14, cr // 6))
            except Exception:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((cx - tw // 2, cy - th // 2), label, fill=(255, 255, 255), font=font,
                      stroke_width=2, stroke_fill=(0, 0, 0))

        if output_path is None:
            output_path = self.render_dir / f"{match_id}_{map_name}.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "PNG", optimize=True)
        return output_path
