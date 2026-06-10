"""
Image-based ring prediction — feed minimap screenshots directly to models.

Two approaches:
  1. CV detection:  Hough Circle Transform extracts ring (x, y, r) from image
                    Then uses existing ML models to predict final zone.

  2. End-to-end CNN: ResNet18 takes minimap image directly,
                    outputs final zone (5-way classification).
                    Trained on synthetic images generated from our 460-game dataset.
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
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from curl_cffi import requests as cffi_requests

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COORD_SPACE = 16384

# ---------------------------------------------------------------------------
# map image loading
# ---------------------------------------------------------------------------

MAP_URLS = {
    "worlds_edge": "https://apexlegendsstatus.com/dgs/mp_rr_desertlands_hu.png",
    "storm_point": "https://apexlegendsstatus.com/dgs/mp_rr_tropic_island_mu2.png",
    "e_district":  "https://apexlegendsstatus.com/dgs/mp_rr_district.png",
}


def _load_map_pil(map_name: str, cache_dir: str | Path = "map_images") -> Image.Image:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    url = MAP_URLS.get(map_name.lower())
    if url is None:
        raise ValueError(f"Unknown map: {map_name}")

    fname = url.split("/")[-1]
    path = cache_dir / fname
    if not path.exists():
        logger.info("Downloading map: %s", url)
        resp = cffi_requests.get(url, impersonate="chrome124", timeout=120)
        path.write_bytes(resp.content)

    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# approach 1: traditional CV ring detection
# ---------------------------------------------------------------------------

def detect_ring_from_image(
    image: np.ndarray | Image.Image | str | Path,
    debug: bool = False,
) -> Optional[Tuple[float, float, float]]:
    """Detect the white ring circle from a minimap screenshot.

    Args:
        image: PIL Image, numpy array (H,W,3), or path to image file.

    Returns:
        (x, y, radius) in 16384-space coordinates, or None if no ring found.
    """
    import cv2

    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise FileNotFoundError(f"Cannot read: {image}")
    elif isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = np.array(image)

    H, W = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to find bright white ring
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Edge detection
    edges = cv2.Canny(thresh, 50, 150)

    # Hough Circle Transform — look for large circles (ring)
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=W // 4,
        param1=50, param2=30,
        minRadius=W // 8, maxRadius=W // 2,
    )

    if circles is None:
        logger.warning("No ring circle detected in image")
        return None

    # Pick the largest circle (ring)
    circles = np.round(circles[0, :]).astype(int)
    best = max(circles, key=lambda c: c[2])
    px, py, pr = best[0], best[1], best[2]

    # Convert image coords to 16384-space
    x = px / W * COORD_SPACE
    y = py / H * COORD_SPACE
    r = pr / W * COORD_SPACE

    logger.info("Detected ring: (%.0f, %.0f) r=%.0f", x, y, r)
    return x, y, r


# ---------------------------------------------------------------------------
# approach 2: end-to-end CNN
# ---------------------------------------------------------------------------

class SyntheticMinimapDataset(Dataset):
    """Generate synthetic minimap images from ring data for CNN training.

    Each sample: a base map with the current ring drawn on it.
    Label: final zone (north/south/east/west/center).
    """

    def __init__(
        self,
        full_rings_path: str | Path,
        map_cache_dir: str = "map_images",
        image_size: int = 224,
        augment: bool = True,
    ):
        self.image_size = image_size
        self.augment = augment

        # Load games
        games = json.loads(Path(full_rings_path).read_text(encoding="utf-8"))
        self.samples: List[dict] = []
        for g in games:
            rings = g["rings"]
            if len(rings) < 4:
                continue
            # 3 stages per game
            for stage_idx in range(3):
                self.samples.append({
                    "map_name": g["map_name"],
                    "ring": rings[stage_idx],
                    "final_ring": rings[3],
                    "stage": stage_idx + 1,
                })

        # Label encoder
        self.le = LabelEncoder()
        all_zones = [_pixel_to_zone(s["final_ring"]["x"], s["final_ring"]["y"])
                     for s in self.samples]
        self.labels = self.le.fit_transform(all_zones)
        self.num_classes = len(self.le.classes_)

        # Cache map images
        logger.info("Loading map images...")
        self.map_images: Dict[str, Image.Image] = {}
        for map_name in set(s["map_name"] for s in self.samples):
            self.map_images[map_name] = _load_map_pil(map_name, map_cache_dir)

        # Transforms
        self.base_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.aug_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info("Dataset: %d samples, %d classes", len(self.samples), self.num_classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        map_img = self.map_images[sample["map_name"]].copy()

        # Draw ring on map
        ring = sample["ring"]
        img = self._draw_ring_pil(map_img, ring)

        transform = self.aug_transform if self.augment else self.base_transform
        tensor = transform(img)
        label = self.labels[idx]
        return tensor, label

    def _draw_ring_pil(self, map_img: Image.Image, ring: dict) -> Image.Image:
        """Draw a semi-transparent white circle on the map image."""
        from PIL import ImageDraw

        W, H = map_img.size
        overlay = Image.new("RGBA", map_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        cx = int(ring["x"] / COORD_SPACE * W)
        cy = int(ring["y"] / COORD_SPACE * H)
        cr = int(ring["r"] / COORD_SPACE * W)

        # Draw filled circle with transparency
        draw.ellipse(
            [cx - cr, cy - cr, cx + cr, cy + cr],
            fill=(255, 255, 255, 60),
            outline=(255, 255, 255, 180),
            width=max(2, cr // 200),
        )

        map_rgba = map_img.convert("RGBA")
        composite = Image.alpha_composite(map_rgba, overlay)
        return composite.convert("RGB")


def _pixel_to_zone(x: float, y: float) -> str:
    cx, cy = x / COORD_SPACE, y / COORD_SPACE
    if 0.35 <= cx <= 0.65 and 0.35 <= cy <= 0.65:
        return "center"
    dx, dy = cx - 0.5, cy - 0.5
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    else:
        return "south" if dy > 0 else "north"


# ---------------------------------------------------------------------------
# CNN model
# ---------------------------------------------------------------------------

class MinimapCNNPredictor(nn.Module):
    """ResNet18 backbone -> final zone classifier (5-way)."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def train_cnn(
    full_rings_path: str | Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.001,
    val_size: float = 0.2,
    image_size: int = 224,
    patience: int = 8,
) -> Tuple[nn.Module, LabelEncoder, dict]:
    """Train end-to-end CNN on synthetic minimap images.

    Returns:
        model, label_encoder, training_history
    """
    dataset = SyntheticMinimapDataset(full_rings_path, image_size=image_size, augment=True)
    le = dataset.le

    # Split
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=42, stratify=dataset.labels
    )

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    model = MinimapCNNPredictor(num_classes=dataset.num_classes).to(DEVICE)
    logger.info("Training CNN on %s, %d train / %d val", DEVICE, len(train_idx), len(val_idx))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_loss += loss_fn(logits, yb).item()
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0:
            logger.info("Epoch %2d: train_loss=%.4f  val_loss=%.4f  val_acc=%.3f",
                        epoch + 1, train_loss, val_loss, val_acc)

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d (best acc=%.3f)", epoch + 1, best_acc)
            break

    if best_state:
        model.load_state_dict(best_state)

    history["best_val_acc"] = best_acc
    return model, le, history


# ---------------------------------------------------------------------------
# prediction from image
# ---------------------------------------------------------------------------

class ImageBasedPredictor:
    """Unified predictor: image in -> zone prediction out.

    Supports two modes:
      - 'cv':   CV detects ring -> ML model predicts zone
      - 'cnn':  End-to-end CNN predicts zone directly from image
    """

    def __init__(
        self,
        mode: str = "cnn",
        cnn_model: Optional[nn.Module] = None,
        cnn_label_encoder: Optional[LabelEncoder] = None,
        rf_predictor=None,
    ):
        self.mode = mode
        self.cnn_model = cnn_model
        self.cnn_le = cnn_label_encoder
        self.rf_predictor = rf_predictor

    def predict(
        self,
        image: np.ndarray | Image.Image | str | Path,
        map_name: Optional[str] = None,
    ) -> Tuple[str, Dict[str, float]]:
        if self.mode == "cnn" and self.cnn_model is not None:
            return self._predict_cnn(image)
        elif self.mode == "cv" and self.rf_predictor is not None:
            return self._predict_cv(image, map_name)
        raise ValueError(f"Invalid mode or missing model: mode={self.mode}")

    def _predict_cnn(
        self, image: np.ndarray | Image.Image | str | Path
    ) -> Tuple[str, Dict[str, float]]:
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        else:
            img = image.convert("RGB")

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        self.cnn_model.eval()
        with torch.no_grad():
            logits = self.cnn_model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        zones = self.cnn_le.classes_
        mapping = {z: float(p) for z, p in zip(zones, probs)}
        for z in ["north", "south", "east", "west", "center"]:
            mapping.setdefault(z, 0.0)

        best = max(mapping, key=mapping.get)
        return best, mapping

    def _predict_cv(
        self, image: np.ndarray | Image.Image | str | Path, map_name: Optional[str]
    ) -> Tuple[str, Dict[str, float]]:
        result = detect_ring_from_image(image)
        if result is None:
            return "center", {z: 0.2 for z in ["north", "south", "east", "west", "center"]}

        x, y, r = result
        from data_models import CircleState
        state = CircleState(map_name or "worlds_edge", x, y, r, stage=1)
        return self.rf_predictor.predict(state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Train CNN
    print("Training End-to-End CNN...")
    model, le, history = train_cnn("collected_data/full_rings.json", epochs=30)

    print(f"\nCNN Best Val Accuracy: {history['best_val_acc']:.3f}")
    print(f"Classes: {list(le.classes_)}")

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "label_encoder": le,
        "history": history,
    }, "models/cnn_minimap.pt")
    print("Model saved to models/cnn_minimap.pt")

    # Quick test on a generated image
    print("\nTesting prediction on synthetic image...")
    dataset = SyntheticMinimapDataset("collected_data/full_rings.json", augment=False)
    img_tensor, label = dataset[42]
    img_pil = T.ToPILImage()((img_tensor * 0.229 + 0.485).clamp(0, 1))  # unnormalize approx

    predictor = ImageBasedPredictor(mode="cnn", cnn_model=model, cnn_label_encoder=le)
    best, probs = predictor.predict(img_pil)
    actual = le.inverse_transform([label])[0]
    hit = "[HIT]" if best == actual else "[MISS]"
    print(f"  Predicted: {best.upper()}  |  Actual: {actual.upper()}  {hit}")
    for z, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"    {z}: {p:.3f}")
