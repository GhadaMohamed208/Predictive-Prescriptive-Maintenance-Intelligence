import streamlit as st

from src import app_state as S
from src import config as C
from src import ui_components as U

p = S.setup_page()
art = S.get_artifacts()
test_m = (art["metrics"] or {}).get("official_test_last_cycle_vs_capped_truth")
name, df, _ = S.dataset()
tbl = S.fleet()
cnt = {s: int((tbl["state"] == s).sum()) for s in C.STATES}

if not st.session_state["entered"]:
    stats = [("Engines loaded", str(df[C.ID_COL].nunique())),
             ("Model", p.metadata["model_name"]),
             ("Test RMSE (cycles)", f"{test_m['rmse']:.1f}" if test_m else "n/a"),
             ("Sensors modelled", str(len(p.sensors)))]
    st.markdown(U.hero_html(stats), unsafe_allow_html=True)
    st.write("")
    c1, c2, c3, _ = st.columns([1.3, 1.3, 1.3, 3])
    if c1.button("Enter Control Room", type="primary", width="stretch"):
        st.session_state["entered"] = True
        st.rerun()
    if c2.button("Open sample dataset", width="stretch"):
        st.session_state["entered"] = True
        st.switch_page("pages/fleet_overview.py")
    if c3.button("Upload data", width="stretch"):
        st.session_state["entered"] = True
        st.switch_page("pages/upload_analyze.py")
    st.caption("Status values above are read from the trained model artifacts and the loaded dataset. "
               "The turbofan graphic is decorative, not live telemetry.")
    st.stop()

U.page_header("Control Room", f"Fleet status for: {name}. Predictions are model estimates at each engine's last observed cycle.",
              "CONTROL ROOM")
cols = st.columns(5)
cols[0].markdown(U.kpi("Engines", str(len(tbl)), note="in the active dataset"), unsafe_allow_html=True)
for c, s in zip(cols[1:4], ["Critical", "Warning", "Healthy"]):
    c.markdown(U.kpi(f"{C.STATE_ICONS[s]} {s}", str(cnt[s]), "engines", "derived from predicted RUL", s), unsafe_allow_html=True)
cols[4].markdown(U.kpi("Median predicted RUL", f"{tbl['rul_pred'].median():.0f}", "cycles", f"capped at {p.cap:.0f}"),
                 unsafe_allow_html=True)
st.write("")
left, right = st.columns([2.2, 1])
with left:
    st.markdown("<div class='panel'><h4>Attention queue — highest priority engines</h4></div>", unsafe_allow_html=True)
    top = tbl.head(10)[["attention_rank", C.ID_COL, "last_cycle", "rul_pred", "state", "anomaly_recent_fraction"]].copy()
    top["anomaly_recent_fraction"] = top["anomaly_recent_fraction"] * 100
    st.dataframe(top, hide_index=True, width="stretch", column_config={
        "attention_rank": st.column_config.NumberColumn("Rank", width="small"),
        C.ID_COL: st.column_config.NumberColumn("Engine"),
        "last_cycle": st.column_config.NumberColumn("Last cycle"),
        "rul_pred": st.column_config.ProgressColumn("Predicted RUL (cycles)", min_value=0, max_value=float(p.cap), format="%.1f"),
        "state": st.column_config.TextColumn("State"),
        "anomaly_recent_fraction": st.column_config.NumberColumn("Recent anomaly flags %", format="%.0f")})
    c1, c2 = st.columns([1, 3])
    pick = c2.selectbox("Open engine", tbl[C.ID_COL].tolist(), label_visibility="collapsed", key="home_pick")
    if c1.button("Open in Engine Intelligence", type="primary"):
        S.goto_engine(pick)
with right:
    st.markdown("<div class='panel'><h4>Model snapshot</h4>"
                f"<b>{p.metadata['model_name']}</b> · v{p.metadata['model_version']}<br>"
                f"<span style='color:#8da0bd;font-size:.85rem'>Engine-grouped CV · official test evaluated once</span></div>",
                unsafe_allow_html=True)
    if test_m:
        st.markdown(U.kpi("Official test MAE / RMSE", f"{test_m['mae']:.1f} / {test_m['rmse']:.1f}", "cycles",
                          f"final cycles of {test_m['n']} test engines, capped truth"), unsafe_allow_html=True)
    st.write("")
    for pg, lab in (("pages/engine_intelligence.py", "Engine Intelligence"), ("pages/what_if_analysis.py", "What-if Analysis"),
                    ("pages/reports.py", "Reports & Exports"), ("pages/upload_analyze.py", "Upload & Analyze")):
        st.page_link(pg, label=lab)
st.write("")
U.note("Derived states (Healthy / Warning / Critical) are configurable decision-support categories computed from predicted RUL; "
       "they are not NASA labels or industrial standards. " + C.DISCLAIMER)
