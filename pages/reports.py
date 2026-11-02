import pandas as pd
import streamlit as st

from src import app_state as S
from src import config as C
from src import report_generator as RG
from src import ui_components as U
from src.engine_analysis import analyze_engine, build_report_info

p = S.setup_page()
name, df, rul_true = S.dataset()
thr = S.thresholds()
art = S.get_artifacts()
U.page_header("Reports & Exports", "Generate professional PDF engineering reports and download CSV results. Reports contain only "
              "computed values; unavailable sections are marked as such.", "REPORTING")
tbl = S.fleet()

t1, t2, t3 = st.tabs(["Engine report (PDF)", "Fleet summary (PDF)", "CSV exports"])
with t1:
    ids = sorted(df[C.ID_COL].unique())
    c1, c2 = st.columns([1, 3])
    e = c1.selectbox("Engine", ids, index=ids.index(st.session_state.get("selected_engine")) if st.session_state.get("selected_engine") in ids else 0, key="rep_eng")
    g = df[df[C.ID_COL] == e].sort_values(C.CYCLE_COL)
    cmin, cmax = int(g[C.CYCLE_COL].min()), int(g[C.CYCLE_COL].max())
    as_of = c2.slider("Report as of cycle", cmin, cmax, cmax, key="rep_cyc") if cmax > cmin else cmax
    if st.button("Generate engine report", type="primary"):
        with st.spinner("Analysing engine and rendering PDF…"):
            res = analyze_engine(p, S.get_anomaly_ref(), S.get_explainer(), g[g[C.CYCLE_COL] <= as_of], thr, st.session_state["sigma_trigger"])
            if "error" in res:
                st.error(res["error"])
            else:
                truth = float(rul_true.loc[e]) if (rul_true is not None and as_of == cmax and e in rul_true.index) else None
                info = build_report_info(res, p, name, art["metrics"], truth)
                for r in info["recs"]:
                    r["status"] = st.session_state["rec_status"].get(r["id"], r["status"])
                st.session_state["pdf_engine"] = (f"engine_{e}_cycle_{as_of}_report.pdf", RG.engine_report_pdf(info))
    pdf = st.session_state.get("pdf_engine")
    if pdf:
        st.success(f"Report ready: {pdf[0]} ({len(pdf[1])/1024:.0f} KB)")
        st.download_button("Download engine report (PDF)", pdf[1], pdf[0], "application/pdf", type="primary")
with t2:
    st.markdown(f"Fleet summary for **{name}** ({len(tbl)} engines).")
    if st.button("Generate fleet report", type="primary"):
        with st.spinner("Rendering PDF…"):
            meta = dict(p.metadata); meta["feature_names"] = p.feature_names
            st.session_state["pdf_fleet"] = RG.fleet_report_pdf({"dataset_name": name, "table": tbl, "counts": S.FA.state_counts(tbl),
                                                                 "thresholds": thr, "model_meta": meta, "metrics": art["metrics"]})
    if st.session_state.get("pdf_fleet"):
        st.download_button("Download fleet report (PDF)", st.session_state["pdf_fleet"], "fleet_summary_report.pdf", "application/pdf", type="primary")
with t3:
    st.download_button("Predictions / fleet table (CSV)", tbl.to_csv(index=False).encode(), "fleet_predictions.csv", "text/csv")
    sc = st.session_state.get("scenarios", [])
    if sc:
        st.download_button("What-if scenario comparisons (CSV)", pd.DataFrame([{k: v for k, v in x.items() if k in
                           ("label", "engine_id", "cycle", "baseline_rul", "scenario_rul", "delta", "baseline_state", "scenario_state")} for x in sc]).to_csv(index=False).encode(),
                           "whatif_scenarios.csv", "text/csv")
    else:
        st.caption("No what-if scenarios have been run in this session.")
    if art["comparison"] is not None:
        st.download_button("Model comparison (CSV)", art["comparison"].to_csv(index=False).encode(), "model_comparison.csv", "text/csv")
    if art["test_pred"] is not None:
        st.download_button("Official test predictions (CSV)", art["test_pred"].to_csv(index=False).encode(), "official_test_predictions.csv", "text/csv")
    if art["validation_log"]:
        import json
        st.download_button("Dataset validation log (JSON)", json.dumps(art["validation_log"], indent=2).encode(), "validation_log.json", "application/json")
