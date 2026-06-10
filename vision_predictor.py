"""
Vision-based ring predictor — trained on ALL ring transitions.

Data sources:
  1. full_rings.json: 460 games, 4 rings each (R1-R4 sequential)
     → Training pairs: R1→R2, R2→R3, R3→R4  (1380 pairs, exact labels)

  2. endring/*.png: Aggregate R5 positions from many games
     → R5 KDE density heatmaps as training targets for R4→R5

Model: U-Net (input: map + current ring, output: next-ring center heatmap)
Prediction: chain R1→R2→R3→R4→R5, render all heatmaps on a single map
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, maximum_filter

from curl_cffi import requests as cffi_requests

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COORD_SPACE = 16384
IMG_SIZE = 256
HEATMAP_SIGMA = 6  # Gaussian sigma for single-point label
R5_HEATMAP_SIGMA = 10  # Wider sigma for R5 density KDE

MAP_URLS = {
    "worlds_edge": "mp_rr_desertlands_hu.png",
    "storm_point": "mp_rr_tropic_island_mu2.png",
    "e_district":  "mp_rr_district.png",
}

RING_SHRINK = [0.50, 0.60, 0.60, 0.50]  # R1→R2, R2→R3, R3→R4, R4→R5


# ---------------------------------------------------------------------------
# training dataset
# ---------------------------------------------------------------------------

class FullRingDataset(Dataset):
    """All ring transition pairs: R1→R2, R2→R3, R3→R4 (from JSON), R4→R5 (from endring)."""

    def __init__(self, full_rings_path: str | Path, cache_dir: str = "map_images"):
        games = json.loads(Path(full_rings_path).read_text(encoding="utf-8"))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.samples: List[dict] = []

        # Load R5 density heatmaps from endring data
        self.r5_heatmaps: Dict[str, np.ndarray] = {}
        for mp in ["storm_point", "worlds_edge", "e_district"]:
            r5_path = Path("collected_data") / f"r5_{mp}.npy"
            if r5_path.exists():
                points = np.load(r5_path)
                hm = self._points_to_heatmap(points)
                self.r5_heatmaps[mp] = hm
                logger.info("Loaded %d R5 points for %s", len(points), mp)

        # R1→R2, R2→R3, R3→R4 from sequential data
        for g in games:
            rings = g["rings"]
            if len(rings) < 4:
                continue
            for i in range(3):
                self.samples.append({
                    "map_name": g["map_name"],
                    "current": rings[i],
                    "next": rings[i + 1],
                    "type": "sequential",  # single-point target
                })

        # R4→R5: use the last ring (R4) from each game as input,
        #         target is the R5 KDE density heatmap
        for g in games:
            if g["map_name"] in self.r5_heatmaps and len(g["rings"]) >= 4:
                self.samples.append({
                    "map_name": g["map_name"],
                    "current": g["rings"][3],  # R4
                    "next": g["rings"][3],  # placeholder, not used for KDE target
                    "type": "r5_kde",  # uses R5 density heatmap
                })

        # Load and cache map images
        self.map_images: Dict[str, Image.Image] = {}
        for map_name in set(s["map_name"] for s in self.samples):
            self.map_images[map_name] = self._load_map(map_name)

        seq_count = sum(1 for s in self.samples if s["type"] == "sequential")
        r5_count = sum(1 for s in self.samples if s["type"] == "r5_kde")
        logger.info("Dataset: %d samples (%d sequential + %d R5-KDE)", len(self.samples), seq_count, r5_count)

    def _points_to_heatmap(self, points: np.ndarray) -> np.ndarray:
        """Convert R5 (x,y) points in 16384-space to a heatmap at IMG_SIZE."""
        hm = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64)
        for px, py in points:
            ix = int(px / COORD_SPACE * IMG_SIZE)
            iy = int(py / COORD_SPACE * IMG_SIZE)
            if 0 <= ix < IMG_SIZE and 0 <= iy < IMG_SIZE:
                hm[iy, ix] += 1.0
        if hm.sum() > 0:
            hm = gaussian_filter(hm, sigma=R5_HEATMAP_SIGMA)
            hm /= hm.max()
        return hm

    def _load_map(self, map_name: str) -> Image.Image:
        url = MAP_URLS.get(map_name.lower())
        if url is None:
            raise ValueError(f"Unknown map: {map_name}")
        path = self.cache_dir / url
        if not path.exists():
            logger.info("Downloading map: %s", url)
            resp = cffi_requests.get(f"https://apexlegendsstatus.com/dgs/{url}",
                                     impersonate="chrome124", timeout=120)
            path.write_bytes(resp.content)
        return Image.open(path).convert("RGB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        map_img = self.map_images[sample["map_name"]]
        current = sample["current"]

        # ---- INPUT: map + current ring ----
        input_img = map_img.copy().resize((IMG_SIZE, IMG_SIZE))
        draw = ImageDraw.Draw(input_img)
        cx = int(current["x"] / COORD_SPACE * IMG_SIZE)
        cy = int(current["y"] / COORD_SPACE * IMG_SIZE)
        cr = int(current["r"] / COORD_SPACE * IMG_SIZE)
        draw.ellipse([cx-cr, cy-cr, cx+cr, cy+cr], outline=(255, 255, 255), width=2)

        overlay = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.ellipse([cx-cr, cy-cr, cx+cr, cy+cr], fill=(255, 255, 255, 30))
        input_rgba = input_img.convert("RGBA")
        input_img = Image.alpha_composite(input_rgba, overlay).convert("RGB")

        input_tensor = T.ToTensor()(input_img)
        input_tensor = T.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(input_tensor)

        # ---- LABEL: next-ring heatmap ----
        if sample["type"] == "sequential":
            next_ring = sample["next"]
            nx = int(next_ring["x"] / COORD_SPACE * IMG_SIZE)
            ny = int(next_ring["y"] / COORD_SPACE * IMG_SIZE)
            yy, xx = torch.meshgrid(
                torch.arange(IMG_SIZE, dtype=torch.float32),
                torch.arange(IMG_SIZE, dtype=torch.float32), indexing="ij",
            )
            heatmap = torch.exp(-((xx - nx)**2 + (yy - ny)**2) / (2 * HEATMAP_SIGMA**2))
            heatmap = heatmap / heatmap.max()
        else:
            # R4→R5: use the R5 KDE density heatmap
            hm = self.r5_heatmaps.get(sample["map_name"])
            if hm is None:
                hm = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            heatmap = torch.from_numpy(hm.astype(np.float32))

        return input_tensor, heatmap.unsqueeze(0)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class RingUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_unet(
    full_rings_path: str = "collected_data/full_rings.json",
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 0.001,
    val_split: float = 0.15,
    patience: int = 12,
    output_path: str = "models/ring_unet.pt",
) -> Tuple[nn.Module, dict]:
    dataset = FullRingDataset(full_rings_path)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = RingUNet(in_ch=3, base_ch=32).to(DEVICE)
    logger.info("Training on %s: %d train / %d val", DEVICE, n_train, n_val)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
        val_loss /= len(val_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss, best_state, no_improve = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %2d: train_loss=%.6f val_loss=%.6f", epoch + 1, train_loss, val_loss)

        if no_improve >= patience:
            logger.info("Early stop at epoch %d (best val_loss=%.6f)", epoch + 1, best_loss)
            break

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_loss
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "history": history}, output_path)
    logger.info("Model saved to %s", output_path)
    return model, history


# ---------------------------------------------------------------------------
# prediction pipeline
# ---------------------------------------------------------------------------

class ChainPredictor:
    """Chain-predict all remaining rings and render heatmaps on the map."""

    def __init__(self, model: nn.Module, cache_dir: str = "map_images"):
        self.model = model.to(DEVICE).eval()
        self.cache_dir = Path(cache_dir)
        self._map_cache: Dict[str, Image.Image] = {}

    def _load_map(self, map_name: str) -> Image.Image:
        if map_name in self._map_cache:
            return self._map_cache[map_name]
        url = MAP_URLS.get(map_name.lower())
        path = self.cache_dir / url
        if not path.exists():
            resp = cffi_requests.get(f"https://apexlegendsstatus.com/dgs/{url}",
                                     impersonate="chrome124", timeout=120)
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(resp.content)
        img = Image.open(path).convert("RGB")
        self._map_cache[map_name] = img
        return img

    def _render_input(self, map_name: str, ring: dict) -> torch.Tensor:
        map_img = self._load_map(map_name).copy().resize((IMG_SIZE, IMG_SIZE))
        draw = ImageDraw.Draw(map_img)
        cx = int(ring["x"] / COORD_SPACE * IMG_SIZE)
        cy = int(ring["y"] / COORD_SPACE * IMG_SIZE)
        cr = int(ring["r"] / COORD_SPACE * IMG_SIZE)
        draw.ellipse([cx-cr, cy-cr, cx+cr, cy+cr], outline=(255, 255, 255), width=2)
        overlay = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (0,0,0,0))
        ImageDraw.Draw(overlay).ellipse([cx-cr, cy-cr, cx+cr, cy+cr], fill=(255,255,255,30))
        img_final = Image.alpha_composite(map_img.convert("RGBA"), overlay).convert("RGB")
        tensor = T.ToTensor()(img_final)
        tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def predict_heatmap(self, map_name: str, ring: dict) -> np.ndarray:
        x = self._render_input(map_name, ring).to(DEVICE)
        hm = self.model(x)[0, 0].cpu().numpy()
        hm = gaussian_filter(hm, sigma=2.0)
        hm = np.clip(hm, 0, None)
        if hm.sum() > 0:
            hm /= hm.sum()
        return hm

    def _heatmap_peak(self, hm: np.ndarray) -> Tuple[int, int]:
        local_max = maximum_filter(hm, size=5) == hm
        y, x = np.unravel_index((hm * local_max).argmax(), hm.shape)
        return x, y

    def predict_all(self, map_name: str, x: float, y: float, r: float, stage: int = 1) -> List[dict]:
        results = []
        current = {"x": x, "y": y, "r": r}
        shrink_idx = stage - 1

        for _ in range(5 - stage):
            hm = self.predict_heatmap(map_name, current)
            px, py = self._heatmap_peak(hm)
            next_x = px / IMG_SIZE * COORD_SPACE
            next_y = py / IMG_SIZE * COORD_SPACE
            next_r = current["r"] * RING_SHRINK[shrink_idx]
            shrink_idx += 1

            results.append({
                "stage": stage + len(results) + 1,
                "x": float(next_x), "y": float(next_y), "r": float(next_r),
                "heatmap": hm,
            })
            current = {"x": next_x, "y": next_y, "r": next_r}

        return results

    def render(self, map_name: str, current_ring: dict, stage: int,
               predictions: List[dict], output_path: str = "prediction.png") -> Path:
        url = MAP_URLS.get(map_name.lower())
        map_img = Image.open(self.cache_dir / url).convert("RGBA")
        W, H = map_img.size

        colors = [(0,255,0,80), (255,255,0,80), (255,165,0,80), (255,50,50,80)]
        peak_colors = [(0,200,0), (200,200,0), (200,130,0), (200,40,40)]

        draw = ImageDraw.Draw(map_img)
        cx = int(current_ring["x"] / COORD_SPACE * W)
        cy = int(current_ring["y"] / COORD_SPACE * H)
        cr = int(current_ring["r"] / COORD_SPACE * W)
        draw.ellipse([cx-cr, cy-cr, cx+cr, cy+cr], outline=(255,255,255), width=4)
        draw.text((cx+8, cy-30), f"R{stage} (current)", fill=(255,255,255))

        overlay = Image.new("RGBA", map_img.size, (0,0,0,0))
        for i, pred in enumerate(predictions):
            if i >= len(colors):
                break
            hm = pred["heatmap"]
            from scipy.ndimage import zoom
            hm_full = zoom(hm, (H/IMG_SIZE, W/IMG_SIZE), order=1)
            hm_full = np.clip(hm_full / max(hm_full.max(), 1e-8), 0, 1)
            colored = Image.new("RGBA", map_img.size, colors[i])
            colored.putalpha(Image.fromarray((hm_full*255).astype(np.uint8), mode="L"))
            overlay = Image.alpha_composite(overlay, colored)

            odraw = ImageDraw.Draw(overlay)
            px, py = int(pred["x"]/COORD_SPACE*W), int(pred["y"]/COORD_SPACE*H)
            pr = int(pred["r"]/COORD_SPACE*W)
            odraw.ellipse([px-pr, py-pr, px+pr, py+pr], outline=peak_colors[i], width=2)
            odraw.text((px+5, py-20), f"R{pred['stage']}", fill=peak_colors[i]+(255,))

        result = Image.alpha_composite(map_img, overlay).convert("RGB")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, "PNG")
        return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Ring U-Net Predictor")
    sub = parser.add_subparsers(dest="mode", required=True)

    tp = sub.add_parser("train", help="Train the U-Net")
    tp.add_argument("--data", default="collected_data/full_rings.json")
    tp.add_argument("--epochs", type=int, default=60)
    tp.add_argument("--output", default="models/ring_unet.pt")

    pp = sub.add_parser("predict", help="Predict ring sequence")
    pp.add_argument("--model", default="models/ring_unet.pt")
    pp.add_argument("--map", required=True, choices=["worlds_edge","storm_point","e_district"])
    pp.add_argument("--x", type=float, required=True)
    pp.add_argument("--y", type=float, required=True)
    pp.add_argument("--radius", type=float, required=True)
    pp.add_argument("--stage", type=int, default=1)
    pp.add_argument("--output", default="prediction.png")

    args = parser.parse_args()

    if args.mode == "train":
        train_unet(args.data, epochs=args.epochs)

    elif args.mode == "predict":
        if Path(args.model).exists():
            model = RingUNet(in_ch=3, base_ch=32).to(DEVICE)
            ckpt = torch.load(args.model, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
        else:
            print(f"No model at {args.model}. Train first.")
            sys.exit(1)

        pred = ChainPredictor(model)
        results = pred.predict_all(args.map, args.x, args.y, args.radius, args.stage)
        current = {"x": args.x, "y": args.y, "r": args.radius}
        path = pred.render(args.map, current, args.stage, results, args.output)
        print(f"Saved: {path}")
        for r in results:
            print(f"  R{r['stage']}: ({r['x']:.0f}, {r['y']:.0f}) r={r['r']:.0f}")
