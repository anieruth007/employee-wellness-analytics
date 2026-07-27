"""ThermaWell dashboard — upload a single FLIR E8 thermal image, get engagement
classification, wellness score, N/C personality proxy, and a plain-language explanation.

On-device only: nothing here is transmitted anywhere. Only an anonymized aggregate team
score would ever be exposed to management (see the aggregate view placeholder below).
"""
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import run_inference

st.set_page_config(page_title="ThermaWell — Employee Wellness Dashboard", layout="centered")

st.title("ThermaWell")
st.caption(
    "Privacy-first engagement monitoring from a single thermal image — "
    "no visible-spectrum camera, no self-report questionnaire."
)

uploaded = st.file_uploader("Upload a FLIR E8 thermal capture", type=["jpg", "jpeg"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    result = None
    try:
        result = run_inference(tmp_path)
    except Exception as e:
        st.error(f"Could not process image: {e}")
    finally:
        os.unlink(tmp_path)

    if result:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded, caption="Uploaded thermal capture")
        with col2:
            st.metric("Engagement", result["engagement"])
            st.metric("Wellness score", f"{result['wellness_score']:.2f}")
            st.progress(result["wellness_score"])

        st.subheader("Personality-proxy signals")
        pcol1, pcol2 = st.columns(2)
        pcol1.metric("N proxy (Neuroticism)", "High" if result["n_proxy"] >= 0.5 else "Low")
        pcol2.metric("C proxy (Conscientiousness)", "High" if result["c_proxy"] >= 0.5 else "Low")

        st.subheader("Explanation")
        st.write(result["explanation"])

        with st.expander("Raw ROI temperatures (°C)"):
            st.json(result["roi_temps_c"])

st.divider()
st.subheader("Team view (management)")
st.caption(
    "Management only ever sees an anonymized aggregate wellness score across the team, "
    "never an individual employee's result — this placeholder will pull from a "
    "locally-aggregated store, not raw per-employee data."
)
st.info("Aggregate team view — not yet wired up.")

st.divider()
st.caption("Employee data stays on-device. Designed for compliance with India's DPDP Act 2023.")
