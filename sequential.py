"""
Sequential model: LSTM-based ring trajectory prediction.

Takes the complete ring pull sequence as input:
    Sequence: [R1, R2, R3]  (3 timesteps)
    Features per ring: [x, y, radius, map_onehot...]
    Predicts: final zone (5-way classification)

Also includes a lightweight Transformer variant for comparison.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from data_models import CircleState, DEFAULT_ZONES

logger = logging.getLogger(__name__)

COORD_SPACE = 16384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------

def _load_sequences(full_rings_path: str | Path) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Load full ring sequences and convert to numpy arrays.

    Returns:
        X: (N, 3, F) where 3 = R1/R2/R3 positions, F = features
        y: (N,) zone labels
        le: LabelEncoder for zones
    """
    path = Path(full_rings_path)
    games = json.loads(path.read_text(encoding="utf-8"))

    # Build map index dynamically
    all_maps = sorted(set(g["map_name"] for g in games))
    map_index = {m: i for i, m in enumerate(all_maps)}
    n_maps = len(all_maps)

    sequences = []
    zones = []

    for g in games:
        rings = g["rings"]
        if len(rings) < 4:
            continue

        # 3 timesteps: R1, R2, R3 (first 3 rings)
        seq = []
        for i in range(3):
            r = rings[i]
            # Features: x, y, r, cx_norm, cy_norm, map_onehot...
            map_vec = [0.0] * n_maps
            map_vec[map_index.get(g["map_name"], 0)] = 1.0

            feats = [
                r["x"] / COORD_SPACE,
                r["y"] / COORD_SPACE,
                r["r"] / COORD_SPACE,
                (r["x"] / COORD_SPACE - 0.5),
                (r["y"] / COORD_SPACE - 0.5),
                *map_vec,
            ]
            seq.append(feats)

        sequences.append(seq)

        # Zone from final ring
        final = rings[3]
        zones.append(_pixel_to_zone(final["x"], final["y"]))

    X = np.array(sequences, dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(zones)
    y = np.array(y, dtype=np.int64)

    return X, y, le


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
# LSTM model
# ---------------------------------------------------------------------------

class RingLSTM(nn.Module):
    """Bidirectional LSTM over ring pull sequence."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, F)
        out, (h_n, c_n) = self.lstm(x)
        # Concatenate final forward + backward hidden states
        last = out[:, -1, :]  # (B, 2*H)
        return self.fc(self.dropout(last))


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------

class RingTransformer(nn.Module):
    """Lightweight Transformer encoder over ring sequence."""

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, F)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        out = self.encoder(x)
        pooled = out.mean(dim=1)  # average pooling over timesteps
        return self.fc(pooled)


# ---------------------------------------------------------------------------
# training wrapper
# ---------------------------------------------------------------------------

@dataclass
class SequentialResult:
    model: nn.Module
    label_encoder: LabelEncoder
    scaler: StandardScaler
    val_accuracy: float
    train_losses: List[float]
    val_losses: List[float]


def train_sequential(
    full_rings_path: str | Path,
    model_type: str = "lstm",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    patience: int = 15,
    val_size: float = 0.2,
) -> SequentialResult:
    """Train an LSTM or Transformer on ring sequences."""
    X, y, le = _load_sequences(full_rings_path)

    # Standardize per-sequence
    scaler = StandardScaler()
    N, T, F = X.shape
    X_flat = X.reshape(-1, F)
    X_flat[:, :3] = scaler.fit_transform(X_flat[:, :3])  # only scale x/y/r
    X = X_flat.reshape(N, T, F)

    # Train / val split
    indices = np.arange(N)
    train_idx, val_idx = train_test_split(indices, test_size=val_size,
                                          random_state=42, stratify=y)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Build model
    input_dim = F
    num_classes = len(le.classes_)

    if model_type == "lstm":
        model = RingLSTM(input_dim, num_classes=num_classes)
    elif model_type == "transformer":
        model = RingTransformer(input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(DEVICE)
    logger.info("Training %s on %s, input_dim=%d, classes=%d",
                model_type.upper(), DEVICE, input_dim, num_classes)
    logger.info("Train: %d, Val: %d", len(train_idx), len(val_idx))

    # Training
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    X_train_t = torch.tensor(X_train).to(DEVICE)
    y_train_t = torch.tensor(y_train).to(DEVICE)
    X_val_t = torch.tensor(X_val).to(DEVICE)
    y_val_t = torch.tensor(y_val).to(DEVICE)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0.0

        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(1, len(X_train_t) // batch_size)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            logger.info("Epoch %3d | train_loss=%.4f val_loss=%.4f val_acc=%.3f",
                        epoch + 1, avg_train_loss, val_loss, val_acc)

        if no_improve >= patience:
            logger.info("Early stopping at epoch %d (best val_acc=%.3f)", epoch + 1, best_val_acc)
            break

    if best_state:
        model.load_state_dict(best_state)

    return SequentialResult(
        model=model, label_encoder=le, scaler=scaler,
        val_accuracy=best_val_acc,
        train_losses=train_losses, val_losses=val_losses,
    )


# ---------------------------------------------------------------------------
# prediction interface
# ---------------------------------------------------------------------------

class SequentialPredictor:
    """Predict final zone from a ring sequence using LSTM/Transformer."""

    def __init__(self, result: SequentialResult):
        self.model = result.model.to(DEVICE)
        self.le = result.label_encoder
        self.scaler = result.scaler

        # Map index from training (inferred from model input dim)
        n_maps = result.model.lstm.input_size - 5 if hasattr(result.model, 'lstm') else result.model.input_proj.in_features - 5
        self.n_maps = n_maps
        self._default_map = 0

    def predict_sequence(self, rings: list[dict], map_name: str) -> Tuple[str, Dict[str, float]]:
        """Predict from a list of ring dicts [{x, y, r}, ...]."""
        seq = []
        for r in rings[:3]:
            feats = [
                r["x"] / COORD_SPACE, r["y"] / COORD_SPACE, r["r"] / COORD_SPACE,
                (r["x"] / COORD_SPACE - 0.5), (r["y"] / COORD_SPACE - 0.5),
            ] + [0.0] * self.n_maps
            seq.append(feats)

        # Pad to 3 timesteps
        while len(seq) < 3:
            seq.append(seq[-1])

        x = np.array([seq], dtype=np.float32)
        x_flat = x.reshape(-1, x.shape[-1])
        x_flat[:, :3] = self.scaler.transform(x_flat[:, :3])
        x = x_flat.reshape(1, 3, -1)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(x).to(DEVICE))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        mapping = {self.le.inverse_transform([i])[0]: float(p)
                   for i, p in enumerate(probs)}
        for zone in DEFAULT_ZONES:
            mapping.setdefault(zone, 0.0)

        best = max(mapping, key=mapping.get)
        return best, mapping

    def predict(self, state: CircleState) -> Tuple[str, Dict[str, float]]:
        """Compatibility wrapper — treats state as single ring."""
        rings = [{"x": state.circle_x, "y": state.circle_y, "r": state.circle_radius}]
        return self.predict_sequence(rings, state.map_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("Training LSTM...")
    result_lstm = train_sequential("collected_data/full_rings.json", model_type="lstm", epochs=100)
    print(f"LSTM val_acc: {result_lstm.val_accuracy:.3f}")

    print("\nTraining Transformer...")
    result_tf = train_sequential("collected_data/full_rings.json", model_type="transformer", epochs=100)
    print(f"Transformer val_acc: {result_tf.val_accuracy:.3f}")

    # Quick test
    predictor = SequentialPredictor(result_lstm)
    rings = [{"x": 7730, "y": 5988, "r": 4825}, {"x": 9144, "y": 3978, "r": 2367}]
    zone, probs = predictor.predict_sequence(rings, "worlds_edge")
    print(f"\nTest LSTM prediction: zone={zone}, probs={ {z: f'{p:.3f}' for z, p in probs.items()} }")
