from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from data_models import DEFAULT_ZONES


ZONE_POSITIONS = {
    "north": (0.5, 0.82),
    "south": (0.5, 0.18),
    "east": (0.82, 0.5),
    "west": (0.18, 0.5),
    "center": (0.5, 0.5),
}


def plot_heatmap(probabilities: Dict[str, float], title: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#101217")
    fig.patch.set_facecolor("#101217")

    xs = []
    ys = []
    sizes = []
    colors = []

    for zone in DEFAULT_ZONES:
        x, y = ZONE_POSITIONS[zone]
        probability = probabilities.get(zone, 0.0)
        xs.append(x)
        ys.append(y)
        sizes.append(1500 * (0.2 + probability))
        colors.append(probability)
        ax.text(x, y, f"{zone}\n{probability:.1%}", ha="center", va="center", color="white", fontsize=12)

    scatter = ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis", alpha=0.85, edgecolors="white", linewidths=1.5)
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Probability")

    ax.set_title(title, color="white", fontsize=16, pad=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#444")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
