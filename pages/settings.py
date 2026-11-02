import json

import streamlit as st

from src import app_state as S
from src import config as C
from src import ui_components as U

p = S.setup_page()
U.page_header("Settings & About", "Configurable decision-support thresholds, data provenance, limitations and references.", "SETTINGS")
st.markdown("#### Derived health-state thresholds")
c1, c2, c3 = st.columns(3)
w = c1.number_input("Warning threshold (cycles)", 1.0, float(p.cap), float(st.session_state["thr_warning"]), 1.0)
cr = c2.number_input("Critical threshold (cycles)", 0.0, float(p.cap), float(st.session_state["thr_critical"]), 1.0)
sg = c3.number_input("Sensor-shift trigger (σ)", 1.0, 10.0, float(st.session_state["sigma_trigger"]), 0.5)
if cr >= w:
    st.error("The critical threshold must be lower than the warning threshold.")
else:
    st.session_state.update({"thr_warning": w, "thr_critical": cr, "sigma_trigger": sg})
    st.caption(f"Healthy: predicted RUL > {w:.0f} · Warning: {cr:.0f} ≤ RUL ≤ {w:.0f} · Critical: RUL < {cr:.0f} cycles. "
               "These are configurable application rules, not NASA labels or industrial standards.")
if st.button("Reset thresholds to defaults"):
    st.session_state.update({"thr_warning": C.DEFAULT_THRESHOLDS["warning"], "thr_critical": C.DEFAULT_THRESHOLDS["critical"], "sigma_trigger": 3.0})
    st.rerun()
if st.button("Clear session analyses, scenarios and action statuses"):
    for k in ("analysis", "scenarios", "last_scenario", "rec_status", "pdf_engine", "pdf_fleet"):
        st.session_state.pop(k, None)
    st.session_state["rec_status"] = {}
    st.rerun()

st.markdown("#### Model & data provenance")
meta = {k: v for k, v in p.metadata.items() if k not in ("feature_names", "feature_config")}
st.json(meta, expanded=False)
log = S.get_artifacts()["validation_log"]
if log:
    with st.expander("Dataset validation log"):
        st.json(log, expanded=False)
st.markdown("#### Limitations")
for t in ("Simulated NASA C-MAPSS FD001 data: single operating condition, single fault mode (HPC degradation). Results do not transfer automatically to real equipment.",
          f"RUL estimates are capped at {p.cap:.0f} cycles and carry measurable error (see Model Performance).",
          "Sensors are named 'Sensor N' only; the supplied documentation does not give a sensor-to-component mapping, so no physical root cause is claimed.",
          "SHAP is model attribution, not causation. What-if edits are hypothetical model inputs, not physical interventions.",
          "The anomaly detector is unsupervised and unvalidated against fault labels.",
          "Recommendation statuses are kept in the browser session only (export CSV to keep them).",
          "No real-time streaming, costs, downtime or maintenance savings are modelled."):
    st.markdown("- " + t)
st.markdown("#### References")
st.markdown("- A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08 (included in `data/docs`).\n"
            "- NASA Prognostics Center of Excellence, C-MAPSS Turbofan Engine Degradation Simulation dataset.\n"
            "- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017 (SHAP).")
U.note(C.DISCLAIMER)
