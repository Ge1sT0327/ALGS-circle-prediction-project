# ALGS Circle Zone Prediction

Predict Apex Legends Global Series (ALGS) circle/ring pull zones using machine learning — supervised, unsupervised, and sequential models.

## Overview

In competitive Apex Legends, knowing where the next ring will pull is a critical strategic advantage. This project:

- **Collects** 460 real ALGS match ring sequences via the apexlegendsstatus.com Ring Guessr API
- **Trains** 6 model types: Baseline, Random Forest (multi-stage), GMM, KDE, LSTM, Transformer
- **Evaluates** with top-1 / top-3 accuracy and 5-fold cross-validation
- **Analyzes** ring-pull patterns using K-Means cluster archetypes
- **Simulates** live games ring-by-ring with real-time predictions
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

# Train multi-stage model
python main.py train --data collected_data/full_rings.json --model rf

# Evaluate all models
python main.py evaluate --data collected_data/full_rings.json --model all

# Simulate a full game
python main.py simulate --game 0 --model rf

# Analyze ring-pull patterns
python main.py analyze --data collected_data/rings.csv --clusters 5

# Predict single state
python main.py predict --map-name worlds_edge --x 5000 --y -6000 --radius 4800 --stage 2 --model rf --data collected_data/rings.csv --heatmap out.png
```

## CLI Reference

```
python main.py {train,predict,evaluate,analyze,simulate,export-sample}
```

| Command | Key Flags | Description |
|---|---|---|
| `train` | `--data PATH --model {baseline,rf,gmm,kde,all}` | Train a model. JSON auto-expands to multi-stage |
| `predict` | `--map-name --x --y --radius --stage [--model] [--heatmap]` | Predict next zone |
| `evaluate` | `--data PATH --model {all,...}` | Evaluate accuracy |
| `analyze` | `--data PATH --clusters N` | K-Means ring-pull archetypes |
| `simulate` | `--game N --model M` | Step through a real game ring-by-ring |
| `export-sample` | `--output PATH` | Export built-in sample to CSV |

## Models

| Model | Type | CV / Val Accuracy | Top-1 | Top-3 | Notes |
|---|---|---|---|---|---|
| **Baseline** | Rule-based | — | 23.3% | 62.1% | No training, heuristic rules |
| **GMM** | Unsupervised | — | 24.8% | 65.9% | Gaussian Mixture, 5 components/map |
| **KDE** | Unsupervised | — | 24.8% | 66.7% | Kernel Density Estimation |
| **Multi-Stage RF** | Supervised | **81.0% (5-Fold CV)** | 97.2% | 100% | Enhanced features, 3 stages × 460 games |
| **LSTM** | Sequential | **95.7% (Val)** | 76.0% | 99.6% | Bidirectional LSTM over ring trajectory |
| **Transformer** | Sequential | **95.7% (Val)** | 76.9% | 98.6% | Transformer encoder over ring trajectory |

Random baseline = 20% (5-class). CV scores are the most honest measure of generalization.

## Multi-Stage Training

Each ALGS game provides 4 rings (R1→R2→R3→Final). Multi-stage training expands each game into 3 labelled samples:

```
R1 → R2  (stage 1)  — predicts final zone from opening circle
R2 → R3  (stage 2)  — predicts final zone from mid-game circle
R3 → R4  (stage 3)  — predicts final zone from late-game circle
```

This gives 1380 training samples (460 × 3) and is the key to the 81% CV accuracy.

## Real-Time Prediction

Interactive tool for predicting during live matches:

```bash
# Interactive mode
python realtime.py interactive

# Quick one-shot
python realtime.py quick --map worlds_edge --x 7730 --y 5988 --radius 4825 --model rf

# Simulate a game
python realtime.py simulate --game 42
```

## Project Structure

```
├── main.py                 Entry point
├── project_cli.py          CLI routing (6 commands)
├── realtime.py             Interactive real-time predictor
├── data_models.py          Core models & Predictor protocol
├── data_io.py              CSV read/write
├── data_sources.py         Data source manifest
├── dataset.py              Built-in sample data
├── feature_engineering.py  ML feature extraction
├── scraper.py              Ring Guessr API scraper
├── training.py             Random Forest (single-stage)
├── multi_stage.py          Multi-stage RF + enhanced features
├── sequential.py           LSTM / Transformer sequence models
├── predictor.py            Rule-based baseline
├── unsupervised.py         GMM / KDE / K-Means models
├── evaluation.py           Accuracy metrics (any predictor)
├── visualization.py        Probability heatmap
├── map_renderer.py         GPU-accelerated map overlay renderer
├── requirements.txt
└── collected_data/
    ├── rings.csv           460-game single-stage dataset
    └── full_rings.json     Full ring sequences (4 rings/game)
```

## Requirements

- Python 3.10+
- numpy, pandas, matplotlib, scikit-learn
- curl_cffi (TLS fingerprinting for API access)
- PyTorch 2.0+ (LSTM/Transformer + GPU map rendering)
- Pillow (map image rendering)

## License

Educational and research purposes. Data sourced from apexlegendsstatus.com.
