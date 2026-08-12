"""ThermaWell dashboard — thermal physiological stress and cognitive load monitoring.

Upload a single thermal image, get continuous Stress Index / Cognitive Load Index /
Wellness Score (0-100 each) plus a measurement confidence score — no classification
labels (see project v2.0.docx: reframed from an engagement classifier to continuous
physiological monitoring).

Accepts a FLIR E8 radiometric JPEG (production) or a raw 16-bit thermal TIFF (test/demo
mode — useful before a physical FLIR E8 is on hand).

On-device only: nothing here is transmitted anywhere.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import run_inference

ROI_COLORS = {"nose_tip": "red", "forehead": "yellow", "periorbital": "cyan", "upper_lip": "magenta"}

st.set_page_config(page_title="ThermaWell — Physiological Monitoring", layout="centered")

st.title("ThermaWell")
st.caption(
    "Privacy-first thermal physiological monitoring — Stress Index, Cognitive Load Index, "
    "and Wellness Score from a single thermal image. No visible-spectrum camera, no "
    "self-report questionnaire, no classification labels."
)


def draw_annotations(gray_image: np.ndarray, bbox, landmarks: dict) -> Image.Image:
    """Grayscale thermal image -> RGB with the detected face bbox and ROI markers drawn on."""
    rgb = np.stack([gray_image] * 3, axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)

    if bbox is not None:
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline="lime", width=2)

    for roi_name, points in landmarks.items():
        color = ROI_COLORS.get(roi_name, "white")
        for px, py in points:
            r = 3
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=color)

    return img


def make_gauge(value: float, title: str, steps: list, bar_color: str, height: int = 250) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar_color},
                "steps": steps,
                "borderwidth": 1,
            },
        )
    )
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=50, b=20))
    return fig


uploaded = st.file_uploader(
    "Upload a thermal capture",
    type=["jpg", "jpeg", "tiff", "tif"],
    help="FLIR E8 radiometric JPEG for real captures, or a raw 16-bit TIFF for testing "
         "without a physical camera (e.g. a Charlotte-ThermalFace sample).",
)

if uploaded is not None:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    result = None
    try:
        with st.spinner("Running inference..."):
            result = run_inference(tmp_path)
    except Exception as e:
        st.error(f"Could not process image: {e}")
    finally:
        os.unlink(tmp_path)

    if result:
        st.subheader("Processed capture")
        annotated = draw_annotations(result["_gray_image"], result["_bbox"], result["_landmarks"])
        st.image(annotated, caption=f"Face bounding box (green) + ROI markers — detector: {result['detector_used']}")

        st.subheader("Scores")
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.plotly_chart(
                make_gauge(
                    result["stress_index"], "Stress Index",
                    steps=[
                        {"range": [0, 40], "color": "#c6f6c6"},
                        {"range": [40, 70], "color": "#fff3b0"},
                        {"range": [70, 100], "color": "#f5a3a3"},
                    ],
                    bar_color="darkred",
                ),
                use_container_width=True,
            )
        with gcol2:
            st.plotly_chart(
                make_gauge(
                    result["cognitive_load_index"], "Cognitive Load Index",
                    steps=[
                        {"range": [0, 33], "color": "#cce5ff"},
                        {"range": [33, 66], "color": "#6699ff"},
                        {"range": [66, 100], "color": "#003399"},
                    ],
                    bar_color="darkblue",
                ),
                use_container_width=True,
            )

        st.plotly_chart(
            make_gauge(
                result["wellness_score"], "Wellness Score",
                steps=[
                    {"range": [0, 25], "color": "#f5a3a3"},
                    {"range": [25, 50], "color": "#ffd9a0"},
                    {"range": [50, 75], "color": "#fff3b0"},
                    {"range": [75, 100], "color": "#c6f6c6"},
                ],
                bar_color="darkgreen",
                height=320,
            ),
            use_container_width=True,
        )

        st.subheader("Measurement confidence")
        st.progress(result["measurement_confidence"] / 100)
        st.caption(f"{result['measurement_confidence']:.0f}% — "
                   f"{'readings are reliable' if result['measurement_confidence'] >= 60 else 'poor thermal image quality, interpret with caution'}")

        st.subheader("ROI temperatures")
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.write("**Raw (°C)**")
            st.table({k: f"{v:.1f}" for k, v in result["roi_temps"].items()})
        with rcol2:
            st.write("**Normalized delta (°C above ambient)**")
            st.table({k: f"{v:.1f}" for k, v in result["normalized_deltas"].items()})
        st.caption(f"Ambient (background) reference for this capture: {result['ambient_temp']:.1f}°C")

        st.subheader("Interpretation")
        interp = result["interpretation"]
        icol1, icol2, icol3 = st.columns(3)
        # Plain markdown, not st.metric — "Needs attention" truncates to "Needs atten…" in
        # st.metric's fixed-width value slot at this column width; metric is meant for
        # numeric+delta display anyway, not categorical text.
        icol1.markdown(f"**Stress level**\n\n{interp['stress_level']}")
        icol2.markdown(f"**Cognitive state**\n\n{interp['cognitive_state']}")
        icol3.markdown(f"**Wellness**\n\n{interp['wellness_flag']}")
        st.info(interp["recommendation"])

        st.subheader("Research grounding")
        st.write("**Stress basis:** " + result["research_grounding"]["stress_basis"])
        st.write("**Cognitive load basis:** " + result["research_grounding"]["cognitive_basis"])

st.divider()
st.caption("Individual data stays on-device. Anonymous team averages only shared with management.")
