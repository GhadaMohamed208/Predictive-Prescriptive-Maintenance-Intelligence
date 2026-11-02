import pandas as pd
import streamlit as st

from src import app_state as S
from src import config as C
from src import recommendation_engine as RE
from src import ui_components as U
from src.engine_analysis import analyze_engine

p = S.setup_page()
name, df, _ = S.dataset()
thr = S.thresholds()
U.page_header("Maintenance Action Center", "Rule-triggered, evidence-aware recommendations with traceable triggers and sources. "
              "Statuses are tracked for this session and can be exported.", "PRESCRIPTIVE")
tbl = S.fleet()
scope = st.radio("Scope", ["Top engines by attention rank", "Specific engine"], horizontal=True)
if scope.startswith("Top"):
    n = st.slider("Number of engines", 1, 15, 5)
    eids = tbl[C.ID_COL].head(n).tolist()
else:
    eids = [st.selectbox("Engine", sorted(df[C.ID_COL].unique()))]


@st.cache_data(show_spinner=False)
def recs_for(name_: str, eid: int, warn: float, crit: float, sigma: float):
    g = df[df[C.ID_COL] == eid]
    r = analyze_engine(p, S.get_anomaly_ref(), S.get_explainer(), g, {"warning": warn, "critical": crit}, sigma)
    return r["recs"], r.get("rul"), r.get("state")


rows = []
with st.spinner("Evaluating rules…"):
    for e in eids:
        recs, rul, state = recs_for(name, int(e), thr["warning"], thr["critical"], st.session_state["sigma_trigger"])
        for r in recs:
            r = dict(r); r["rul"], r["state"] = rul, state
            r["status"] = st.session_state["rec_status"].get(r["id"], r["status"])
            rows.append(r)
if not rows:
    U.note("No rules were triggered for the selected scope.")
    st.stop()
R = pd.DataFrame(rows)
cnt = R["priority"].value_counts()
k = st.columns(5)
for c, pr in zip(k, ["Critical", "High", "Medium", "Low", "Info"]):
    c.markdown(U.kpi(pr, str(int(cnt.get(pr, 0))), "actions"), unsafe_allow_html=True)
st.write("")
st.markdown("##### Action register (edit status inline)")
view = R[["id", "engine_id", "priority", "title", "evidence_strength", "status", "rul"]].copy()
edited = st.data_editor(view, hide_index=True, width="stretch", height=360, disabled=[c for c in view.columns if c != "status"],
                        column_config={"id": "ID", "engine_id": "Engine", "priority": "Priority", "title": "Recommendation",
                                       "evidence_strength": "Evidence", "rul": st.column_config.NumberColumn("Pred. RUL", format="%.1f"),
                                       "status": st.column_config.SelectboxColumn("Status", options=RE.STATUSES, required=True)})
for _, r in edited.iterrows():
    st.session_state["rec_status"][r["id"]] = r["status"]
R["status"] = R["id"].map(lambda i: st.session_state["rec_status"].get(i, "Pending"))
st.download_button("Export action register (CSV)", R.drop(columns=["sensors"]).assign(sensors=R["sensors"].map("; ".join)).to_csv(index=False).encode(),
                   "maintenance_actions.csv", "text/csv")

st.markdown("##### Details")
for _, r in R.iterrows():
    with st.expander(f"Engine {r['engine_id']} · {r['priority']} · {r['title']} [{r['status']}]"):
        st.markdown(f"**Action:** {r['action']}")
        st.markdown(f"**Trigger:** {r['trigger']}  \n**Evidence:** {r['evidence']}")
        if r["sensors"]:
            st.markdown("**Relevant sensors:** " + ", ".join(r["sensors"]))
        st.markdown(f"**Evidence strength:** {r['evidence_strength']} · **Finding type:** {r['finding_type']}  \n**Source:** {r['source']}")
        st.markdown(f"**Limitations:** {r['limitation']}  \n**Verification:** {r['verification']}")
with st.expander("Rules registry and definitions"):
    st.dataframe(pd.DataFrame([{"Rule": r.rule_id, "Title": r.title, "Trigger": r.trigger_text, "Priority": r.priority, "Source": r.source}
                               for r in RE.RULES]), hide_index=True, width="stretch")
    st.markdown("**Evidence strength:** " + " · ".join(f"*{k}* — {v}" for k, v in RE.EVIDENCE_DEFINITIONS.items()))
    U.note("Rules are decision-support conventions. None maps a sensor to a physical component, and none claims that an action will "
           "extend life by a quantified amount.")
