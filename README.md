# ALGS Circle Zone Prediction

Predict Apex Legends Global Series (ALGS) circle/ring pull zones using machine learning.

## Overview

In competitive Apex Legends (ALGS), predicting where the next ring will pull is a critical strategic advantage. This project:

- **Collects** real ALGS match ring data via the apexlegendsstatus.com Ring Guessr API
- **Trains** a Random Forest classifier to predict the final zone (north/south/east/west/center)
- **Evaluates** predictions with top-1 and top-3 accuracy metrics
- **Visualizes** zone probability heatmaps

## Data Source

Ring data is scraped from the [apexlegendsstatus.com](https://apexlegendsstatus.com/algs/ring-guessr) Ring Guessr API, which serves real ALGS match ring positions. The API provides ring center coordinates (x, y) and radius for each round, plus the final zone.

Maps supported: World's Edge, Storm Point, Broken Moon, E-District, Kings Canyon, Olympus.

## Installation

```bash
git clone https://github.com/Ge1sT0327/ALGS-circle-prediction-project.git
cd ALGS-circle-prediction-project
pip install -r requirements.txt
```

## Usage

### Collect ring data

```bash
# Collect 500 ALGS ring samples (saved to collected_data/rings.csv)
python scraper.py
```

Or from Python:

```python
from scraper import scrape_to_csv
scrape_to_csv(target=200, output_path="collected_data/rings.csv")
```

### Train a model

```bash
# Train on collected data
python main.py train --data collected_data/rings.csv

# Or train on built-in sample data
python main.py train
```

### Predict next zone

```bash
python main.py predict \
  --map-name worlds_edge \
  --x 500 --y -800 \
  --radius 3200 \
  --stage 2 \
  --heatmap prediction.png
```

### Evaluate

```bash
python main.py evaluate --data collected_data/rings.csv
```

## Project Structure

| File | Purpose |
|------|---------|
| `scraper.py` | Ring data collection from Ring Guessr API |
| `data_models.py` | Dataclass models (CircleRecord, CircleState, ZonePrediction) |
| `data_io.py` | CSV import/export for ring records |
| `data_sources.py` | Data source manifest and JSON I/O |
| `dataset.py` | Sample dataset for quick testing |
| `feature_engineering.py` | Feature extraction (one-hot maps, stage buckets) |
| `training.py` | Random Forest model training |
| `predictor.py` | Rule-based baseline predictor |
| `evaluation.py` | Accuracy metrics (top-1, top-3) |
| `visualization.py` | Zone probability heatmap |
| `project_cli.py` | CLI argument parsing and command routing |
| `main.py` | Entry point |

## Model

The primary model is a Random Forest classifier (`RandomForestClassifier`) with:
- 200 estimators, max depth 8, balanced class weights
- Features: normalized circle center (x, y), radius, stage, map one-hot encoding, stage bucket
- Target: 5-way zone classification (north/south/east/west/center)

A simple rule-based baseline (`SimpleCirclePredictor`) is also provided for comparison.

## Requirements

- Python 3.10+
- numpy, pandas, matplotlib, scikit-learn
- curl_cffi (TLS fingerprint impersonation for API access)

## License

This project is for educational and research purposes. Data sourced from apexlegendsstatus.com.
