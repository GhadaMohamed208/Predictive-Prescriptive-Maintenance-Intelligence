import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import anomaly_detection as AD
from src import app_state as S
from src import config as C
from src import ui_components as U

p = S.setup_page()
train, test, rul_t = S.load_data()
name, df, _ = S.dataset()
thr = S.thresholds()
U.page_header("Sensor & Degradation Analysis", "Exploratory analysis of the loaded data, per-engine degradation behaviour and a separate "
              "unsupervised anomaly view.", "DEGRADATION ANALYSIS")
tab_eda, tab_deg, tab_anom = st.tabs(["Dataset EDA", "Engine degradation", "Anomaly analysis"])

# ------------------------------------------------------------------ EDA
with tab_eda:
    tr_rul = S.get_artifacts()["oof"]
    k = st.columns(4)
    k[0].markdown(U.kpi("Train engines / rows", f"{train[C.ID_COL].nunique()} / {len(train):,}"), unsafe_allow_html=True)
    k[1].markdown(U.kpi("Test engines / rows", f"{test[C.ID_COL].nunique()} / {len(test):,}"), unsafe_allow_html=True)
    k[2].markdown(U.kpi("Columns", "26", note="unit, cycle, 3 settings, 21 sensors"), unsafe_allow_html=True)
    k[3].markdown(U.kpi("Modelled sensors", str(len(p.sensors)), note=f"excluded: {', '.join(p.metadata['excluded_sensors'])}"),
                  unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    life = train.groupby(C.ID_COL)[C.CYCLE_COL].max()
    with c1:
        fig = go.Figure(go.Histogram(x=life, nbinsx=20, marker_color=U.SERIES[0], marker_line=dict(color="#0d1626", width=1)))
        fig.update_xaxes(title="Cycles to failure"); fig.update_yaxes(title="Engines")
        st.plotly_chart(U.style_fig(fig, "Run-to-failure life of training engines (between-engine variability)", 320, False), width="stretch")
    with c2:
        raw = tr_rul["rul_raw"] if tr_rul is not None else None
        if raw is not None:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=raw, nbinsx=40, name="Raw RUL", marker_color=U.SERIES[2], opacity=.8))
            fig.add_trace(go.Histogram(x=raw.clip(upper=p.cap), nbinsx=40, name=f"Capped at {p.cap:.0f}", marker_color=U.SERIES[0], opacity=.7))
            fig.update_layout(barmode="overlay"); fig.update_xaxes(title="RUL (cycles)"); fig.update_yaxes(title="Rows")
            st.plotly_chart(U.style_fig(fig, "Training RUL distribution (raw vs. capped modelling target)", 320), width="stretch")
    corr = train[p.sensors].corr()
    fig = go.Figure(go.Heatmap(z=corr.values, x=[C.sensor_label(s) for s in corr.columns], y=[C.sensor_label(s) for s in corr.index],
                               zmin=-1, zmax=1, colorscale=[[0, "#d95926"], [.5, "#1a2234"], [1, "#3987e5"]],
                               hovertemplate="%{y} × %{x}<br>ρ=%{z:.2f}<extra></extra>", colorbar=dict(title="Pearson r")))
    st.plotly_chart(U.style_fig(fig, "Sensor correlation (training data, modelled sensors)", 520, False), width="stretch")
    c1, c2 = st.columns(2)
    sd = c1.selectbox("Sensor distribution", p.sensors, format_func=C.sensor_label, key="eda_sd")
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=train[sd], name="Train", marker_color=U.SERIES[0], opacity=.75, histnorm="probability density"))
        fig.add_trace(go.Histogram(x=test[sd], name="Test", marker_color=U.SERIES[1], opacity=.6, histnorm="probability density"))
        fig.update_layout(barmode="overlay"); fig.update_xaxes(title=C.sensor_label(sd)); fig.update_yaxes(title="Density")
        st.plotly_chart(U.style_fig(fig, f"{C.sensor_label(sd)} distribution", 320), width="stretch")
    with c2:
        # early-life vs late-life comparison on the training data
        t2 = train.copy()
        t2["rul_raw"] = t2.groupby(C.ID_COL)[C.CYCLE_COL].transform("max") - t2[C.CYCLE_COL]
        early, late = t2[t2[C.CYCLE_COL] <= 30][sd], t2[t2["rul_raw"] <= 30][sd]
        fig = go.Figure()
        fig.add_trace(go.Box(y=early, name="First 30 cycles", marker_color=U.SERIES[2]))
        fig.add_trace(go.Box(y=late, name="Last 30 cycles before failure", marker_color=U.SERIES[1]))
        fig.update_yaxes(title=C.sensor_label(sd))
        st.plotly_chart(U.style_fig(fig, "Early-life vs. late-life behaviour (training engines)", 320, False), width="stretch")
    with st.expander("Column descriptions & dataset facts"):
        st.markdown("`unit` engine id · `cycle` operating cycle · `op1–op3` operational settings · `s1–s21` sensor measurements. "
                    "Physical meaning/units of sensors are not documented in the supplied files and are not assumed here. "
                    "FD001: one operating condition, one fault mode (HPC degradation) per the dataset readme.")
        art = S.get_artifacts()["validation_log"]
        if art:
            st.json(art["subset_identity"], expanded=False)
            for n in art.get("notes", []):
                st.caption("• " + n)

# ------------------------------------------------------------------ degradation
with tab_deg:
    ids = sorted(df[C.ID_COL].unique())
    c1, c2, c3 = st.columns([1, 2, 1])
    e = c1.selectbox("Engine", ids, key="deg_eng", index=ids.index(st.session_state.get("selected_engine")) if st.session_state.get("selected_engine") in ids else 0)
    sens = c2.multiselect("Sensors", p.sensors, default=p.sensors[1:3], format_func=C.sensor_label, key="deg_s")
    g = df[df[C.ID_COL] == e].sort_values(C.CYCLE_COL).reset_index(drop=True)
    cr = c3.slider("Cycle range", int(g[C.CYCLE_COL].min()), int(g[C.CYCLE_COL].max()),
                   (int(g[C.CYCLE_COL].min()), int(g[C.CYCLE_COL].max())), key="deg_cr") if len(g) > 1 else (1, 1)
    gg = g[g[C.CYCLE_COL].between(*cr)]
    fig = go.Figure()
    for i, s in enumerate(sens):
        col = U.SERIES[i % 8]
        base = g[s].iloc[:C.BASELINE_CYCLES].mean()
        fig.add_trace(go.Scatter(x=gg[C.CYCLE_COL], y=gg[s] - base, name=f"{C.sensor_label(s)} − baseline", mode="lines",
                                 line=dict(color=col, width=1), opacity=.5))
        fig.add_trace(go.Scatter(x=gg[C.CYCLE_COL], y=(gg[s] - base).rolling(10, min_periods=1).mean(),
                                 name=f"{C.sensor_label(s)} (10-cycle mean)", mode="lines", line=dict(color=col, width=2.5)))
    fig.add_hline(y=0, line=dict(color="#8da0bd", width=1))
    fig.update_xaxes(title="Operating cycle"); fig.update_yaxes(title="Change vs. engine's own early baseline (sensor units)")
    st.plotly_chart(U.style_fig(fig, f"Engine {e}: change relative to the baseline (mean of first {C.BASELINE_CYCLES} cycles)", 380), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        # reference: median and 10-90% band of training engines, aligned by time-to-failure
        s1 = st.selectbox("Sensor vs. fleet reference", p.sensors, format_func=C.sensor_label, key="deg_ref")
        t2 = train.copy()
        t2["ttf"] = t2.groupby(C.ID_COL)[C.CYCLE_COL].transform("max") - t2[C.CYCLE_COL]
        t2 = t2[t2["ttf"] <= 200]
        q = t2.groupby("ttf")[s1].quantile([.1, .5, .9]).unstack()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=q.index, y=q[.9], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=q.index, y=q[.1], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(57,135,229,.2)",
                                 name="Training engines 10–90%"))
        fig.add_trace(go.Scatter(x=q.index, y=q[.5], mode="lines", name="Training median", line=dict(color=U.SERIES[0], width=2)))
        if name.startswith("Training") or name.startswith("Official"):
            pass
        fig.update_xaxes(title="Cycles before failure (training engines, true RUL)", autorange="reversed")
        fig.update_yaxes(title=C.sensor_label(s1))
        st.plotly_chart(U.style_fig(fig, "Reference behaviour of training engines approaching failure", 340), width="stretch")
        st.caption("Reference band uses TRUE time-to-failure of the training engines; it describes how sensors typically behave, "
                   "not a prediction for the selected engine.")
    with c2:
        rh = p.predict_rows(g).rename(columns={C.CYCLE_COL: "cycle"})
        m = g.merge(rh, left_on=C.CYCLE_COL, right_on="cycle")
        fig = go.Figure(go.Scatter(x=m[s1], y=m["rul_pred"], mode="markers",
                                   marker=dict(color=m[C.CYCLE_COL], colorscale="Blues", size=6, showscale=True,
                                               colorbar=dict(title="Cycle"), line=dict(color="#0d1626", width=.5)),
                                   hovertemplate="cycle %{marker.color}<br>sensor %{x:.4g}<br>RUL %{y:.1f}<extra></extra>"))
        fig.update_xaxes(title=C.sensor_label(s1)); fig.update_yaxes(title="Model-estimated RUL (cycles)")
        st.plotly_chart(U.style_fig(fig, f"Engine {e}: {C.sensor_label(s1)} vs. estimated RUL", 340, False), width="stretch")

# ------------------------------------------------------------------ anomaly
with tab_anom:
    ref = S.get_anomaly_ref()
    st.markdown("**Method:** Isolation Forest fitted on early-life behaviour (cycles "
                f"{ref['ref_cycles'][0]}–{ref['ref_cycles'][1]}) of the training engines; input is each sensor's 10-cycle mean minus the "
                f"engine's own early baseline. A cycle is flagged when its score exceeds the {ref['quantile']*100:.0f}th percentile of the "
                "reference scores. This view is **separate from the supervised RUL model**; an anomaly is not automatically a failure.")
    ids = sorted(df[C.ID_COL].unique())
    e2 = st.selectbox("Engine", ids, key="anom_eng")
    g = df[df[C.ID_COL] == e2].sort_values(C.CYCLE_COL)
    sc = AD.score(ref, g)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sc[C.CYCLE_COL], y=sc["anomaly_score"], mode="lines", name="Anomaly score", line=dict(color=U.SERIES[0], width=2)))
    fl = sc[sc["anomaly_flag"]]
    fig.add_trace(go.Scatter(x=fl[C.CYCLE_COL], y=fl["anomaly_score"], mode="markers", name="Flagged", marker=dict(color="#ef4444", size=7, symbol="diamond")))
    fig.add_hline(y=ref["threshold"], line=dict(color="#f59e0b", dash="dash"), annotation_text="threshold", annotation_font=dict(color="#f59e0b"))
    fig.update_xaxes(title="Operating cycle"); fig.update_yaxes(title="Isolation Forest anomaly score")
    st.plotly_chart(U.style_fig(fig, f"Engine {e2}: anomaly score over cycles (causal)", 340), width="stretch")
    summ = AD.engine_summary(sc)
    k = st.columns(3)
    k[0].markdown(U.kpi("Recent flagged", f"{summ['recent_flag_fraction']*100:.0f}%", note=f"last {summ['recent_n']} cycles"), unsafe_allow_html=True)
    k[1].markdown(U.kpi("First flagged cycle", str(summ["first_flag_cycle"] or "—")), unsafe_allow_html=True)
    k[2].markdown(U.kpi("Total flagged cycles", str(summ["n_flagged_total"])), unsafe_allow_html=True)
    m = (S.get_artifacts()["metrics"] or {}).get("anomaly_detector_sanity")
    if m:
        st.caption(f"Sanity check on training data (no fault labels exist): flag rate {m['flag_rate_rul>125']*100:.1f}% when true RUL>125, "
                   f"{m['flag_rate_rul_60_125']*100:.1f}% for 60–125, {m['flag_rate_rul<=60']*100:.1f}% for ≤60; Spearman ρ between score and "
                   f"elapsed life = {m['spearman_score_vs_elapsed_life_train']:.2f}. This shows the score tracks progression; it is not a fault-classification accuracy.")
