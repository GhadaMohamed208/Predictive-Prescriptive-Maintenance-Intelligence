import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import app_state as S
from src import config as C
from src import ui_components as U

p = S.setup_page()
name, df, rul_true = S.dataset()
thr = S.thresholds()
tbl = S.fleet()
U.page_header("Fleet Overview", f"{name}: simulated engines of the supplied dataset (not a real operating fleet). "
              "Predicted RUL at each engine's last observed cycle.", "FLEET INTELLIGENCE")
if name.startswith("Training"):
    U.note("Training engines run to failure, so their last cycle has true RUL = 0 and is expected to be Critical. "
           "Use the Engine Intelligence 'as-of cycle' control for earlier points.", warn=True)

cnt = S.FA.state_counts(tbl)
cols = st.columns(5)
cols[0].markdown(U.kpi("Engines", str(len(tbl))), unsafe_allow_html=True)
for c, s in zip(cols[1:4], ["Critical", "Warning", "Healthy"]):
    c.markdown(U.kpi(f"{C.STATE_ICONS[s]} {s}", str(cnt[s]), "engines", cls=s), unsafe_allow_html=True)
cols[4].markdown(U.kpi("Persistent anomalies", str(int(tbl["anomaly_persistent"].sum())), "engines",
                       "≥50% of last 10 cycles flagged"), unsafe_allow_html=True)
st.write("")

# ---- filters
f1, f2, f3, f4 = st.columns([1.4, 1, 1.4, 1.4])
states = f1.multiselect("Risk state", C.STATES, default=C.STATES)
eid = f2.text_input("Engine ID contains", "")
rmax = float(p.cap)
rr = f3.slider("Predicted RUL range (cycles)", 0.0, rmax, (0.0, rmax))
cmin, cmax = int(tbl["last_cycle"].min()), int(tbl["last_cycle"].max())
cr = f4.slider("Last-cycle range", cmin, cmax, (cmin, cmax)) if cmin < cmax else (cmin, cmax)
flt = tbl[tbl["state"].isin(states) & tbl["rul_pred"].between(*rr) & tbl["last_cycle"].between(*cr)]
if eid.strip():
    flt = flt[flt[C.ID_COL].astype(str).str.contains(eid.strip())]

c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Bar(x=C.STATES, y=[cnt[s] for s in C.STATES], marker_color=[C.STATE_COLORS[s] for s in C.STATES],
                           text=[cnt[s] for s in C.STATES], textposition="outside",
                           hovertemplate="%{x}: %{y} engines<extra></extra>", width=.5))
    fig.update_yaxes(title="Engines"); fig.update_xaxes(title="Derived state")
    st.plotly_chart(U.style_fig(fig, "Risk-state distribution", 330, False), width="stretch")
with c2:
    fig = go.Figure(go.Histogram(x=tbl["rul_pred"], nbinsx=25, marker_color=U.SERIES[0], marker_line=dict(color="#0d1626", width=1),
                                 hovertemplate="RUL %{x}: %{y} engines<extra></extra>"))
    for lab, v, col in (("Warning", thr["warning"], C.STATE_COLORS["Warning"]), ("Critical", thr["critical"], C.STATE_COLORS["Critical"])):
        fig.add_vline(x=v, line=dict(color=col, dash="dash", width=1), annotation_text=lab, annotation_font=dict(color=col, size=10))
    fig.update_xaxes(title="Predicted RUL (cycles)"); fig.update_yaxes(title="Engines")
    st.plotly_chart(U.style_fig(fig, "Distribution of predicted RUL", 330, False), width="stretch")

st.markdown(f"##### Engine table ({len(flt)} of {len(tbl)} shown)")
show = flt[["attention_rank", C.ID_COL, "last_cycle", "n_cycles", "rul_pred", "state", "at_cap", "anomaly_recent_fraction",
            "ood_fraction"] + (["rul_true_official"] if "rul_true_official" in flt else [])].copy()
show["anomaly_recent_fraction"] *= 100
show["ood_fraction"] *= 100
st.dataframe(show, hide_index=True, width="stretch", height=360, column_config={
    "attention_rank": "Rank", C.ID_COL: "Engine", "last_cycle": "Last cycle", "n_cycles": "Cycles observed",
    "rul_pred": st.column_config.ProgressColumn("Predicted RUL", min_value=0, max_value=float(p.cap), format="%.1f"),
    "state": "State", "at_cap": st.column_config.CheckboxColumn("At cap"),
    "anomaly_recent_fraction": st.column_config.NumberColumn("Anomaly flags % (last 10)", format="%.0f"),
    "ood_fraction": st.column_config.NumberColumn("Out-of-range features %", format="%.0f"),
    "rul_true_official": st.column_config.NumberColumn("Official true RUL (eval only)", format="%.0f")})
d1, d2 = st.columns([1, 4])
d1.download_button("Export filtered table (CSV)", flt.to_csv(index=False).encode(), "fleet_filtered.csv", "text/csv")
with d2:
    U.note("Ranking: derived state (Critical first) → lower predicted RUL → persistent anomaly flag. No costs, downtime or savings are "
           "modelled. 'Official true RUL' is shown only for evaluation of the official test set.")

st.markdown("##### Drill-down and trajectory comparison")
sel = st.multiselect("Compare engines (up to 6)", flt[C.ID_COL].tolist(), default=flt[C.ID_COL].head(3).tolist(), max_selections=6)
if sel:
    t1, t2 = st.tabs(["Estimated RUL history", "Sensor trajectory"])
    with t1:
        fig = go.Figure()
        for i, e in enumerate(sel):
            h = p.predict_rows(df[df[C.ID_COL] == e])
            fig.add_trace(go.Scatter(x=h[C.CYCLE_COL], y=h["rul_pred"], mode="lines", name=f"Engine {e}",
                                     line=dict(color=U.SERIES[i % 8], width=2),
                                     hovertemplate=f"Engine {e}<br>cycle %{{x}}<br>RUL %{{y:.1f}}<extra></extra>"))
        U.add_thresholds(fig, thr)
        fig.update_xaxes(title="Operating cycle"); fig.update_yaxes(title="Estimated RUL (cycles)")
        st.plotly_chart(U.style_fig(fig, "Model-estimated RUL per cycle (causal: each point uses data up to that cycle)", 400), width="stretch")
    with t2:
        sens = st.selectbox("Sensor", p.sensors, format_func=C.sensor_label, key="fleet_sensor")
        fig = go.Figure()
        for i, e in enumerate(sel):
            g = df[df[C.ID_COL] == e].sort_values(C.CYCLE_COL)
            fig.add_trace(go.Scatter(x=g[C.CYCLE_COL], y=g[sens], mode="lines", name=f"Engine {e}",
                                     line=dict(color=U.SERIES[i % 8], width=1.6)))
        fig.update_xaxes(title="Operating cycle"); fig.update_yaxes(title=C.sensor_label(sens))
        st.plotly_chart(U.style_fig(fig, f"{C.sensor_label(sens)} trajectories", 400), width="stretch")
c1, c2 = st.columns([3, 1])
pick = c1.selectbox("Open engine", flt[C.ID_COL].tolist() or tbl[C.ID_COL].tolist(), key="fleet_pick")
if c2.button("Open in Engine Intelligence", type="primary"):
    S.goto_engine(pick)
