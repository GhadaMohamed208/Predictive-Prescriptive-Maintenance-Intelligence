"""Industrial Predictive & Prescriptive Maintenance Intelligence Platform - Streamlit entry point.

Run:  streamlit run app.py
"""
import streamlit as st

from src import config as C

st.set_page_config(page_title=C.PRODUCT_SHORT, page_icon=":material/precision_manufacturing:",
                   layout="wide", initial_sidebar_state="expanded")

PAGES = [
    st.Page("pages/home.py", title="Control Room", icon=":material/dashboard:", default=True),
    st.Page("pages/fleet_overview.py", title="Fleet Overview", icon=":material/groups:"),
    st.Page("pages/engine_intelligence.py", title="Engine Intelligence", icon=":material/monitor_heart:"),
    st.Page("pages/sensor_analysis.py", title="Sensor & Degradation", icon=":material/show_chart:"),
    st.Page("pages/explainable_ai.py", title="Explainable AI", icon=":material/psychology:"),
    st.Page("pages/maintenance_center.py", title="Maintenance Actions", icon=":material/build:"),
    st.Page("pages/what_if_analysis.py", title="What-if Analysis", icon=":material/tune:"),
    st.Page("pages/upload_analyze.py", title="Upload & Analyze", icon=":material/upload_file:"),
    st.Page("pages/reports.py", title="Reports & Exports", icon=":material/description:"),
    st.Page("pages/model_performance.py", title="Model Performance", icon=":material/analytics:"),
    st.Page("pages/settings.py", title="Settings & About", icon=":material/settings:"),
]
with st.sidebar:
    st.markdown("<div style='font:700 .95rem ui-monospace,monospace;letter-spacing:1.5px;color:#22d3ee'>"
                "◈ MAINTENANCE INTELLIGENCE</div><div style='font-size:.72rem;color:#8da0bd;margin:2px 0 6px'>"
                "Decision support · NASA C-MAPSS FD001</div>", unsafe_allow_html=True)
st.navigation(PAGES).run()
