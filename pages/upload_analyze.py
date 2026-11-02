import pandas as pd
import streamlit as st

from src import app_state as S
from src import config as C
from src import ui_components as U
from src import upload_mapping as UM
from src.data_loader import DataFormatError, read_rul_file, read_uploaded_bytes
from src.data_validation import validate_trajectories
from src.evaluate_models import regression_metrics

p = S.setup_page()
thr = S.thresholds()
_, test, _ = S.load_data()
U.page_header("Upload & Analyze", "Bring your own engine trajectories. The file is validated, mapped and checked for compatibility before "
              "any prediction is made. Original uploads are never modified.", "CSV WORKFLOW")

with st.expander("Supported input modes", expanded=False):
    st.markdown("**Mode A – compatible trajectories:** CSV with engine id, cycle and the 14 model sensors (Sensor 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21). "
                "Predicts RUL at the latest cycle; cannot be verified without ground truth.  \n"
                "**Mode B – evaluation data:** Mode A + a ground-truth RUL file (NASA `RUL_FD00x` format, one value per engine at its last cycle) → metrics computed.  \n"
                "**Mode C – unsupported:** missing sensors, or values far outside the training distribution (different machine/units). No prediction is produced; "
                "the reason and the required next steps are explained.")
st.download_button("Download example CSV template (2 real FD001 test engines)", UM.example_template(test).to_csv(index=False).encode(),
                   "cmapss_template.csv", "text/csv")

# ---- 1. select file
f = st.file_uploader(f"1 · Select a CSV / whitespace text file (max {C.MAX_UPLOAD_MB} MB)", type=["csv", "txt", "tsv"], key="upl_file")
f_rul = st.file_uploader("Optional · ground-truth RUL file (one value per engine, ordered by engine id)", type=["txt", "csv"], key="upl_rul")
if f is None:
    U.note("Waiting for a file. Nothing has been uploaded or analysed yet.")
    st.stop()
st.caption(f"File: **{f.name}** · {f.size/1024:.1f} KB · status: received")
try:
    raw = f.getvalue()
    original = read_uploaded_bytes(raw)
except DataFormatError as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

# ---- 2. preview
st.markdown("##### 2 · Schema inspection")
st.dataframe(original.head(8), hide_index=True, width="stretch")
st.caption(f"{len(original):,} rows × {original.shape[1]} columns")

# ---- 3. mapping
st.markdown("##### 3 · Column mapping (auto-detected; adjust if needed)")
guess = UM.infer_columns(original)
opts = ["— not mapped —"] + list(original.columns)


def _idx(v):
    return opts.index(v) if v in opts else 0


m1, m2 = st.columns(2)
unit = m1.selectbox("Engine / unit id column", opts, index=_idx(guess["unit"]), key="map_unit")
cyc = m2.selectbox("Cycle column", opts, index=_idx(guess["cycle"]), key="map_cycle")
smap = {}
cols = st.columns(4)
for i, s in enumerate(C.MODEL_SENSORS):
    choice = cols[i % 4].selectbox(C.sensor_label(s), opts, index=_idx(guess["sensors"][s]), key=f"map_{s}")
    smap[s] = None if choice == opts[0] else choice
missing = [s for s, v in smap.items() if v is None]
if unit == opts[0] or cyc == opts[0]:
    st.error("Engine id and cycle columns must be mapped.")
    st.stop()

# ---- 4. validate
st.markdown("##### 4 · Data-quality report")
try:
    mapped = UM.apply_mapping(original, unit, cyc, {k: v for k, v in smap.items() if v})
except ValueError as e:
    st.error(str(e)); st.stop()
use_sensors = [s for s in C.MODEL_SENSORS if smap[s]]
report = validate_trajectories(mapped, sensors=use_sensors) if not missing else validate_trajectories(mapped, sensors=use_sensors)
comp = UM.compatibility(mapped, p.reference, use_sensors) if report.ok and use_sensors else None
has_truth = f_rul is not None
mode, why = UM.classify_mode(report, comp, missing, has_truth)
k = st.columns(4)
k[0].markdown(U.kpi("Validation", report.status.title(), note=f"{len(report.errors)} error(s) · {len(report.warnings)} warning(s)"), unsafe_allow_html=True)
k[1].markdown(U.kpi("Engines", str(report.stats.get("engines", "—"))), unsafe_allow_html=True)
k[2].markdown(U.kpi("Rows", f"{report.stats.get('rows', len(mapped)):,}"), unsafe_allow_html=True)
k[3].markdown(U.kpi("Input mode", {"A": "A · Compatible", "B": "B · Evaluation", "C": "C · Unsupported", "invalid": "Invalid"}[mode]), unsafe_allow_html=True)
rf = report.to_frame()
if len(rf):
    st.dataframe(rf, hide_index=True, width="stretch")
st.download_button("Download validation results (CSV)", rf.to_csv(index=False).encode(), "validation_results.csv", "text/csv")
if comp and comp["warn_sensors"]:
    U.note("Sensors with >20% of values outside the training range: " + ", ".join(C.sensor_label(s) for s in comp["warn_sensors"])
           + ". Check units/scaling.", warn=True)
if mode in ("C", "invalid"):
    st.error(why if why else "Cannot proceed.")
    st.stop()

# ---- Mode B truth
rul_true = None
if has_truth:
    try:
        rul_true = read_rul_file(f_rul)
        ids = sorted(mapped[C.ID_COL].unique())
        if len(rul_true) != len(ids):
            st.error(f"RUL file has {len(rul_true)} values but the data has {len(ids)} engines. Mode B requires one value per engine.")
            st.stop()
        rul_true.index = ids
    except Exception as e:
        st.error(f"Could not read the RUL file: {e}"); st.stop()

# ---- 5. confirm & run
st.markdown("##### 5 · Confirm and run")
st.checkbox("I confirm the mapping and validation results above are correct.", key="upl_confirm")
if st.button("Run analysis", type="primary", disabled=not st.session_state.get("upl_confirm", False)):
    with st.spinner("Running model inference…"):
        st.session_state["upload"] = {"ready": True, "df": mapped.sort_values([C.ID_COL, C.CYCLE_COL]).reset_index(drop=True),
                                      "name": f.name, "mode": mode, "report": report.to_dict(), "rul_true": rul_true,
                                      "original_bytes": raw}
        st.session_state["_pending_source"] = S.SRC_UPLOAD
        st.rerun()

up = st.session_state.get("upload")
if up and up.get("ready") and up["name"] == f.name:
    st.markdown("##### 6 · Results")
    tbl = S.fleet(S.SRC_UPLOAD)
    cnt = S.FA.state_counts(tbl)
    k = st.columns(4)
    for c, s in zip(k[:3], ["Critical", "Warning", "Healthy"]):
        c.markdown(U.kpi(f"{C.STATE_ICONS[s]} {s}", str(cnt[s]), "engines", cls=s), unsafe_allow_html=True)
    k[3].markdown(U.kpi("Median predicted RUL", f"{tbl['rul_pred'].median():.0f}", "cycles"), unsafe_allow_html=True)
    show = tbl[["attention_rank", C.ID_COL, "last_cycle", "n_cycles", "rul_pred", "state", "ood_fraction"]
               + (["rul_true_official"] if "rul_true_official" in tbl else [])]
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "rul_pred": st.column_config.ProgressColumn("Predicted RUL", min_value=0, max_value=float(p.cap), format="%.1f"),
        "ood_fraction": st.column_config.NumberColumn("Out-of-range features", format="percent"),
        "rul_true_official": st.column_config.NumberColumn("Provided true RUL", format="%.0f")})
    if up["mode"] == "B" and "rul_true_official" in tbl:
        y = tbl["rul_true_official"].clip(upper=p.cap)
        m = regression_metrics(y, tbl["rul_pred"])
        st.markdown(f"**Mode B evaluation (capped truth, n={m['n']}):** MAE {m['mae']:.2f} · RMSE {m['rmse']:.2f} · R² {m['r2']:.3f} · NASA score {m['nasa_score']:.0f}")
    else:
        U.note("No ground truth was provided: these predictions cannot be directly verified against known RUL.")
    d1, d2, d3 = st.columns([1.3, 1.3, 3])
    d1.download_button("Download predictions (CSV)", tbl.to_csv(index=False).encode(), "uploaded_predictions.csv", "text/csv")
    if d2.button("Open in Fleet Overview", type="primary"):
        st.switch_page("pages/fleet_overview.py")
    st.caption("The uploaded dataset is now the active data source: Engine Intelligence, Explainable AI, Maintenance Actions and Reports use it.")
