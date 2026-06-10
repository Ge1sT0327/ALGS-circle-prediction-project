# ALGS Circle Zone Prediction

Predict Apex Legends Global Series (ALGS) circle/ring pull zones using machine learning — supervised and unsupervised.

## Overview

In competitive Apex Legends, knowing where the next ring will pull is a critical strategic advantage. This project:

- **Collects** real ALGS match ring data (460 games) via the apexlegendsstatus.com Ring Guessr API
- **Trains** 4 model types: Baseline (rule-based), Random Forest, GMM, KDE
- **Evaluates** with top-1 / top-3 accuracy and cross-validation
- **Analyzes** ring-pull patterns using K-Means cluster archetypes
- **Visualizes** zone probability heatmaps
- **Renders** ring overlays on full-resolution map images (GPU accelerated)

## Installation

```bash
git clone https://github.com/Ge1sT0327/ALGS-circle-prediction-project.git
cd ALGS-circle-prediction-project
pip install -r requirements.txt
```

## Quick Start

```bash
# Collect ring data (~460 ALGS games)
python scraper.py

# Evaluate all models
python main.py evaluate --data collected_data/rings.csv --model all

# Predict a zone
python main.py predict --map-name worlds_edge --x 5000 --y -6000 --radius 4800 --stage 2 --model rf --data collected_data/rings.csv --heatmap out.png

# Discover ring-pull patterns
python main.py analyze --data collected_data/rings.csv --clusters 5

# Render ring overlays on map images
python -c "
from map_renderer import MapRendererGPU
import json
r = MapRendererGPU()
games = json.load(open('collected_data/full_rings.json'))
for g in games:
    r.render_game(g['match_id'], g['map_name'], g['rings'])
print(f'Done: {len(games)} images in {r.render_dir}/')
"
```

## CLI Reference

```
python main.py {train,predict,evaluate,analyze,export-sample}
```

| Command | Flags | Description |
|---|---|---|
| `train` | `--data CSV --model {baseline,rf,gmm,kde,all}` | Train a model (default: rf) |
| `predict` | `--map-name --x --y --radius --stage [--model] [--data] [--heatmap]` | Predict next zone |
| `evaluate` | `--data CSV --model {baseline,rf,gmm,kde,all}` | Evaluate accuracy |
| `analyze` | `--data CSV --clusters N` | K-Means ring-pull archetypes |
| `export-sample` | `--output PATH` | Export built-in sample to CSV |

## Models

| Model | Type | Top-1* | Top-3* | Notes |
|---|---|---|---|---|
| **Baseline** | Rule-based heuristic | 26.7% | 64.8% | No training needed |
| **Random Forest** | Supervised | 59% (CV) | — | 5-fold CV, 200 trees, max_depth=8 |
| **GMM** | Unsupervised | 24.8% | 67.6% | Gaussian Mixture, 5 components/map |
| **KDE** | Unsupervised | 17.6% | 60.4% | Kernel Density Estimation |

*Top-1/3 on 460-game training set. RF should use CV score for realistic estimate.  
Random baseline = 20% (5 classes).

## Project Structure

```
├── main.py                 Entry point
├── project_cli.py          CLI routing
├── data_models.py          Core models & constants
├── data_io.py              CSV read/write
├── data_sources.py         Data source manifest
├── dataset.py              Built-in sample data
├── feature_engineering.py  ML feature extraction
├── scraper.py              Ring Guessr API scraper
├── training.py             Random Forest training
├── predictor.py            Rule-based baseline
├── unsupervised.py         GMM / KDE / K-Means models
├── evaluation.py           Accuracy metrics
├── visualization.py        Probability heatmap
├── map_renderer.py         GPU-accelerated map overlay renderer
├── requirements.txt
└── collected_data/
    ├── rings.csv           460-game training dataset
    └── full_rings.json     Full ring sequences for rendering
```

## Data

- **Source**: [apexlegendsstatus.com](https://apexlegendsstatus.com/algs/ring-guessr) Ring Guessr API
- **Size**: 460 unique ALGS games across 3 competitive maps
- **Format**: Ring center (x, y) and radius in 16384x16384 pixel coordinate space
- **Maps**: World's Edge, Storm Point, E-District
- **Zones**: north / south / east / west / center

## Requirements

- Python 3.10+
- numpy, pandas, matplotlib, scikit-learn
- curl_cffi (TLS fingerprinting for API access)
- PyTorch (GPU map rendering, optional)
- Pillow (map image rendering)

## License

Educational and research purposes. Data sourced from apexlegendsstatus.com.
