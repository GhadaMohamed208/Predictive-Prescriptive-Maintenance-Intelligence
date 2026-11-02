"""Streamlit state & cached resources shared by all pages (models are loaded once)."""
from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from . import config as C
from . import explainability as EX
from . import fleet_analysis as FA
from .data_loader import load_fd001
from .inference import ArtifactsMissingError, Predictor, artifacts_available

SRC_TEST = "Official test set (FD001)"
SRC_TRAIN = "Training set (run-to-failure)"
SRC_UPLOAD = "Uploaded data"


@st.cache_resource(show_spinner=False)
def get_predictor() -> Predictor:
    return Predictor.load()


@st.cache_resource(show_spinner=False)
def get_anomaly_ref():
    return joblib.load(C.ANOMALY_PATH) if C.ANOMALY_PATH.exists() else None


@st.cache_resource(show_spinner=False)
def get_explainer():
    p = get_predictor()
    g = EX.load_global(C.SHAP_GLOBAL_PATH)
    try:
        bg = g["X"].sample(min(200, len(g["X"])), random_state=C.RANDOM_SEED) if g else None
        expl, kind = EX.make_explainer(p.model, background=bg)
        return expl
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_global_shap():
    return EX.load_global(C.SHAP_GLOBAL_PATH)


@st.cache_data(show_spinner=False)
def load_data():
    return load_fd001()


@st.cache_data(show_spinner=False)
def get_artifacts() -> dict:
    out: dict = {}
    out["metrics"] = json.loads(C.METRICS_PATH.read_text()) if C.METRICS_PATH.exists() else None
    out["comparison"] = pd.read_csv(C.COMPARISON_PATH) if C.COMPARISON_PATH.exists() else None
    out["ablation"] = pd.read_csv(C.ABLATION_PATH) if C.ABLATION_PATH.exists() else None
    out["oof"] = pd.read_csv(C.OOF_PATH) if C.OOF_PATH.exists() else None
    out["test_pred"] = pd.read_csv(C.TEST_PRED_PATH) if C.TEST_PRED_PATH.exists() else None
    out["validation_log"] = json.loads(C.VALIDATION_LOG_PATH.read_text()) if C.VALIDATION_LOG_PATH.exists() else None
    return out


def require_artifacts() -> bool:
    """Render a setup message and return False when the model has not been trained."""
    if artifacts_available():
        return True
    st.error("Trained model artifacts were not found.")
    st.markdown("Train and evaluate the models first (takes a few minutes):")
    st.code("python -m src.train_models", language="bash")
    st.caption("No placeholder model or metrics are shipped; the app only shows results from real runs.")
    return False


# ----------------------------------------------------------------- settings / state
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("entered", False)
    ss.setdefault("thr_warning", C.DEFAULT_THRESHOLDS["warning"])
    ss.setdefault("thr_critical", C.DEFAULT_THRESHOLDS["critical"])
    ss.setdefault("sigma_trigger", 3.0)
    ss.setdefault("source", SRC_TEST)
    ss.setdefault("upload", None)          # dict(df, name, mode, report, rul_true, id)
    ss.setdefault("rec_status", {})        # recommendation id -> status
    ss.setdefault("analysis_done", {})     # (source, engine, as_of) -> bool (progressive reveal played)
    ss.setdefault("selected_engine", None)


def thresholds() -> dict:
    return {"warning": float(st.session_state["thr_warning"]), "critical": float(st.session_state["thr_critical"])}


def dataset(source: str | None = None):
    """Return (name, df, rul_true Series|None) for the active data source."""
    source = source or st.session_state["source"]
    train, test, rul = load_data()
    if source == SRC_TEST:
        return source, test, rul
    if source == SRC_TRAIN:
        return source, train, None
    up = st.session_state.get("upload")
    if up is not None and up.get("ready"):
        return SRC_UPLOAD + f" ({up['name']})", up["df"], up.get("rul_true")
    return SRC_TEST, test, rul


def available_sources() -> list[str]:
    s = [SRC_TEST, SRC_TRAIN]
    up = st.session_state.get("upload")
    if up is not None and up.get("ready"):
        s.append(SRC_UPLOAD)
    return s


@st.cache_data(show_spinner=False)
def _fleet_cached(key: str, df_hash: int, warn: float, crit: float, df: pd.DataFrame, rul_true) -> pd.DataFrame:
    return FA.fleet_table(get_predictor(), df, {"warning": warn, "critical": crit}, get_anomaly_ref(), rul_true)


def fleet(source: str | None = None) -> pd.DataFrame:
    name, df, rul = dataset(source)
    t = thresholds()
    h = int(pd.util.hash_pandas_object(df[[C.ID_COL, C.CYCLE_COL]], index=False).sum() % (2 ** 31))
    return _fleet_cached(name, h, t["warning"], t["critical"], df, rul)


def inject_sidebar_status() -> None:
    p = get_predictor()
    pend = st.session_state.pop("_pending_source", None)
    if pend:
        st.session_state["source"] = pend
    with st.sidebar:
        st.markdown("---")
        st.caption("ACTIVE DATA SOURCE")
        srcs = available_sources()
        if st.session_state["source"] not in srcs:
            st.session_state["source"] = SRC_TEST
        st.selectbox("Data source", srcs, key="source", label_visibility="collapsed")
        t = thresholds()
        st.caption(f"Model: **{p.metadata['model_name']}** · v{p.metadata['model_version']}")
        st.caption(f"States: Healthy > {t['warning']:.0f} · Warning {t['critical']:.0f}–{t['warning']:.0f} · "
                   f"Critical < {t['critical']:.0f} cycles")


def setup_page():
    """Common prelude for every page: state, CSS, artifact check, sidebar status. Returns Predictor."""
    from . import ui_components as U
    init_state()
    U.inject_css()
    if not require_artifacts():
        st.stop()
    inject_sidebar_status()
    return get_predictor()


def goto_engine(engine_id: int) -> None:
    st.session_state["selected_engine"] = int(engine_id)
    st.switch_page("pages/engine_intelligence.py")
