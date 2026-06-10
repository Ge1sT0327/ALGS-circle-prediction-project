"""
Gradio Web UI for ALGS Circle Prediction.

Features:
  - Select map, set ring position with sliders or click on map
  - See real-time heatmap prediction for all subsequent rings
  - Download result
"""

from __future__ import annotations

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

import torch
from vision_predictor import RingUNet, ChainPredictor, COORD_SPACE, MAP_URLS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stage descriptions and typical radii
STAGE_INFO = {
    1: {"name": "Ring 1 (开局)", "radius": 4900},
    2: {"name": "Ring 2 (二圈)", "radius": 2400},
    3: {"name": "Ring 3 (三圈)", "radius": 1400},
    4: {"name": "Ring 4 (四圈)", "radius": 700},
    5: {"name": "Ring 5 (五圈/决赛)", "radius": 350},
}

MAP_NAMES = ["storm_point", "worlds_edge", "e_district"]
MAP_LABELS = ["Storm Point", "World's Edge", "E-District"]

# ---------------------------------------------------------------------------
# load model once
# ---------------------------------------------------------------------------

predictor = None

def load_model():
    global predictor
    if predictor is not None:
        return
    model_path = Path("models/ring_unet.pt")
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Train first: python vision_predictor.py train")
    model = RingUNet(in_ch=3, base_ch=32).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    predictor = ChainPredictor(model)
    print("Model loaded.")

# ---------------------------------------------------------------------------
# prediction function
# ---------------------------------------------------------------------------

def predict(map_name: str, stage: int, x: float, y: float, radius: float) -> Image.Image:
    load_model()

    results = predictor.predict_all(map_name, x, y, radius, stage)
    current = {"x": x, "y": y, "r": radius}

    # Render to a temp file
    output = Path("web_output.png")
    predictor.render(map_name, current, stage, results, output)

    # Return the image
    return Image.open(output)

# ---------------------------------------------------------------------------
# auto-set radius when stage changes
# ---------------------------------------------------------------------------

def on_stage_change(stage: int):
    return STAGE_INFO[stage]["radius"]

# ---------------------------------------------------------------------------
# map preview with ring drawn
# ---------------------------------------------------------------------------

def preview_map(map_name: str, x: float, y: float, radius: float, stage: int):
    """Show the base map with current ring drawn on it."""
    url = MAP_URLS[map_name.lower()]
    path = Path("map_images") / url
    if not path.exists():
        return None
    img = Image.open(path).convert("RGBA")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    cx = int(x / COORD_SPACE * W)
    cy = int(y / COORD_SPACE * H)
    cr = int(radius / COORD_SPACE * W)
    draw.ellipse([cx-cr, cy-cr, cx+cr, cy+cr], outline=(255, 255, 255), width=5)
    draw.text((cx+8, cy-30), f"R{stage}", fill=(255, 255, 255))
    return img.resize((512, 512))

# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

PRESETS = {
    "Storm Point - 开局圈": ("storm_point", 1, 8000, 8000, 4900),
    "Storm Point - 后期圈": ("storm_point", 4, 7500, 6500, 700),
    "World's Edge - 开局圈": ("worlds_edge", 1, 8000, 8000, 4900),
    "World's Edge - 后期圈": ("worlds_edge", 4, 8000, 8000, 700),
    "E-District - 开局圈": ("e_district", 1, 8000, 8000, 4900),
}

def apply_preset(preset_name: str):
    if preset_name in PRESETS:
        mp, st, x, y, r = PRESETS[preset_name]
        return mp, st, x, y, r
    return "storm_point", 1, 8000, 8000, 4900

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="ALGS Circle Predictor", theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # ALGS Circle Zone Predictor

    选择地图和当前圈位置，模型预测所有后续圈的热力图。
    - 绿色 = R2/R3, 黄色 = R3/R4, 橙色 = R4/R5, 红色 = R5/梅花桩
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 输入参数")

            preset = gr.Dropdown(
                choices=list(PRESETS.keys()), label="预设场景",
            )

            map_selector = gr.Dropdown(
                choices=MAP_NAMES, value="storm_point", label="地图",
            )

            stage = gr.Slider(1, 5, value=1, step=1, label="当前圈阶段")
            stage_desc = gr.Textbox(value="Ring 1", interactive=False, label="阶段描述")

            x_slider = gr.Slider(0, 16384, value=8000, step=100, label="X 坐标")
            y_slider = gr.Slider(0, 16384, value=8000, step=100, label="Y 坐标")
            radius_slider = gr.Slider(100, 10000, value=4900, step=50, label="半径")

            predict_btn = gr.Button("预测", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### 当前圈（地图预览）")
            preview = gr.Image(value=None, label="", height=512)

            gr.Markdown("### 预测结果")
            result = gr.Image(value=None, label="", height=512)

    # ---- events ----

    # Preset
    preset.change(
        apply_preset,
        inputs=[preset],
        outputs=[map_selector, stage, x_slider, y_slider, radius_slider],
    )

    # Stage change → auto radius
    stage.change(
        lambda s: (STAGE_INFO[s]["radius"], STAGE_INFO[s]["name"]),
        inputs=[stage], outputs=[radius_slider, stage_desc],
    )

    # Preview update on any slider change
    for inp in [map_selector, x_slider, y_slider, radius_slider, stage]:
        inp.change(
            preview_map,
            inputs=[map_selector, x_slider, y_slider, radius_slider, stage],
            outputs=[preview],
        )

    # Predict
    predict_btn.click(
        predict,
        inputs=[map_selector, stage, x_slider, y_slider, radius_slider],
        outputs=[result],
    )

    # Load preview on start
    app.load(
        preview_map,
        inputs=[map_selector, x_slider, y_slider, radius_slider, stage],
        outputs=[preview],
    )


if __name__ == "__main__":
    load_model()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
