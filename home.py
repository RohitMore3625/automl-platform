import streamlit as st
import pandas as pd
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# CARD HELPER
# =====================================================
def card():
    return st.container(border=True)

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
defaults = {
    "df": None,
    "target": None,
    "problem_type": None,
    "preference": "Balanced",
    "best_pipeline": None,
    "best_pipeline_problem_type": None,
    "best_clustering_model": None,
    "best_time_series_model": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<h1>🤖 AutoML Intelligence Platform</h1>
<p style="color:#9ca3af; font-size:18px">
Build, Compare & Explain ML Models — Made By Rohit More
</p>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# DATA INGESTION + INTELLIGENCE
# =====================================================
left, right = st.columns([1, 2])

# -----------------------------
# DATA INGESTION
# -----------------------------
with left:
    with card():
        st.subheader("📂 Data Ingestion")

        file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx"]
        )

        if file:
            try:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)

                # New dataset invalidates any previously selected
                # target / problem type / trained model, otherwise
                # stale state can crash the downstream pages.
                if st.session_state.df is None or not df.equals(st.session_state.df):
                    st.session_state.df = df
                    st.session_state.target = None
                    st.session_state.problem_type = None
                    st.session_state.best_pipeline = None
                    st.session_state.best_pipeline_problem_type = None
                    st.session_state.best_clustering_model = None
                    st.session_state.best_time_series_model = None

                st.success("Dataset loaded successfully")

            except Exception as e:
                st.error(str(e))

# -----------------------------
# DATA INTELLIGENCE
# -----------------------------
with right:
    with card():
        st.subheader("🧠 Data Intelligence")

        df = st.session_state.df

        c1, c2, c3, c4 = st.columns(4)

        if df is None:
            c1.metric("Rows", "-")
            c2.metric("Columns", "-")
            c3.metric("Missing %", "-")
            c4.metric("Duplicates", "-")
        else:
            rows, cols = df.shape
            missing = (df.isnull().sum().sum() / (rows * cols)) * 100 if rows * cols > 0 else 0.0
            dup = df.duplicated().sum()

            c1.metric("Rows", rows)
            c2.metric("Columns", cols)
            c3.metric("Missing %", f"{missing:.2f}")
            c4.metric("Duplicates", dup)

            with st.expander("Preview"):
                st.dataframe(df.head(), use_container_width=True)

st.markdown("##")

# =====================================================
# AUTOML STRATEGY
# =====================================================
with card():
    st.subheader("⚙️ AutoML Strategy")

    pref = st.radio(
        "Optimization Preference",
        ["Fast", "Balanced", "High Accuracy"],
        horizontal=True,
        index=["Fast", "Balanced", "High Accuracy"].index(st.session_state.preference)
    )

    st.session_state.preference = pref

st.markdown("##")

# =====================================================
# TARGET & PROBLEM DETECTION
# =====================================================
if st.session_state.df is not None:
    with card():
        st.subheader("🎯 Target & Problem Detection")

        df = st.session_state.df
        cols = df.columns.tolist()

        options = ["-- None (Clustering) --"] + cols
        current_target = st.session_state.target
        default_index = options.index(current_target) if current_target in cols else 0

        target = st.selectbox(
            "Select Target Column",
            options,
            index=default_index
        )

        if target != "-- None (Clustering) --":
            st.session_state.target = target
        else:
            st.session_state.target = None

        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
        has_datetime = len(datetime_cols) > 0

        # Detection
        # NOTE: Time Series requires BOTH a datetime column AND a target.
        # Previously this forced "Time Series" whenever a datetime column
        # existed even with no target selected, which made Clustering
        # (which requires no target) impossible to reach on such datasets.
        if st.session_state.target is None:
            problem = "Clustering"
        elif has_datetime:
            problem = "Time Series"
        elif df[target].dtype == "object" or df[target].nunique() <= 20:
            problem = "Classification"
        else:
            problem = "Regression"

        st.session_state.problem_type = problem

        c1, c2 = st.columns(2)
        c1.metric("Detected Problem", problem)
        c2.metric("Strategy", st.session_state.preference)

st.markdown("##")

# =====================================================
# ACTION CENTER (NAVIGATION)
# =====================================================
with card():
    st.subheader("🚀 Proceed")

    if st.session_state.df is None:
        st.info("Upload a dataset above to get started.")

    if st.button("▶ Go to AutoML Training", use_container_width=True, disabled=st.session_state.df is None):

        if st.session_state.problem_type == "Regression":
            st.switch_page("pages/2_Regression.py")

        elif st.session_state.problem_type == "Classification":
            st.switch_page("pages/3_Classification.py")

        elif st.session_state.problem_type == "Clustering":
            st.switch_page("pages/4_Clustering.py")

        elif st.session_state.problem_type == "Time Series":
            st.switch_page("pages/5_TimeSeries.py")

        else:
            st.error("Unable to detect problem type")
