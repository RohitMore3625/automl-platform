import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Clustering AutoML",
    page_icon="🧩",
    layout="wide"
)

def card():
    return st.container(border=True)

# =====================================================
# SESSION STATE INITIALIZATION (in case this page is
# opened directly, without visiting Home first)
# =====================================================
defaults = {
    "df": None,
    "target": None,
    "problem_type": None,
    "preference": "Balanced",
    "best_clustering_model": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================================
# VALIDATION
# =====================================================
if st.session_state.df is None:
    st.error("No dataset found. Go back to Home page.")
    st.stop()

if st.session_state.target is not None:
    st.warning("Target selected. Clustering requires no target.")
    st.stop()

# =====================================================
# LOAD DATA
# =====================================================
df = st.session_state.df.copy()
preference = st.session_state.preference

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<h1>🧩 Clustering AutoML</h1>
<p style="color:#9ca3af">Unsupervised clustering and model selection</p>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# DATA OVERVIEW
# =====================================================
with card():
    st.subheader("📂 Dataset Overview")
    c1, c2 = st.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])

    with st.expander("Preview Data"):
        st.dataframe(df.head(), use_container_width=True)

# =====================================================
# NUMERIC FEATURE SELECTION
# =====================================================
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

if len(num_cols) < 2:
    st.error("Clustering requires at least 2 numeric columns.")
    st.stop()

X = df[num_cols]

# =====================================================
# PREPROCESSING
# =====================================================
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_scaled = pipeline.fit_transform(X)

# =====================================================
# MODEL ZOO
# =====================================================
# NOTE: n_clusters can't exceed the number of samples, and clustering
# generally needs more than a handful of rows to be meaningful.
n_samples = X_scaled.shape[0]

models = {}
if n_samples >= 3:
    models["KMeans (k=3)"] = KMeans(n_clusters=3, random_state=42, n_init=10)
    models["Agglomerative (k=3)"] = AgglomerativeClustering(n_clusters=3)
if n_samples >= 5:
    models["KMeans (k=5)"] = KMeans(n_clusters=5, random_state=42, n_init=10)
if preference == "High Accuracy" and n_samples >= 8:
    models["KMeans (k=8)"] = KMeans(n_clusters=8, random_state=42, n_init=10)

if not models:
    st.error("Not enough rows to run clustering (need at least 3).")
    st.stop()

# =====================================================
# RUN AUTOML
# =====================================================
with card():
    st.subheader("🚀 Run Clustering AutoML")

    run = st.button("Run Clustering", use_container_width=True)

    if run:
        results = []

        with st.spinner("Running clustering models..."):
            for name, model in models.items():
                try:
                    labels = model.fit_predict(X_scaled)

                    if len(set(labels)) <= 1:
                        score = np.nan
                    else:
                        score = silhouette_score(X_scaled, labels)

                    results.append({
                        "Model": name,
                        "Clusters": len(set(labels)),
                        "Silhouette Score": score
                    })

                except Exception as e:
                    results.append({
                        "Model": name,
                        "Clusters": "Error",
                        "Silhouette Score": np.nan
                    })

        results_df = pd.DataFrame(results)

        st.subheader("📊 Clustering Results")
        st.dataframe(results_df, use_container_width=True)

        # =================================================
        # SELECT BEST MODEL
        # =================================================
        # NOTE: if every model errored out or produced a single cluster,
        # every score is NaN. Previously the app would still pick a
        # "best" model from this and crash trying to re-fit it.
        valid_results = results_df.dropna(subset=["Silhouette Score"])

        if valid_results.empty:
            st.error(
                "No model produced a valid clustering result. "
                "Try a different dataset or preference."
            )
            st.stop()

        best_row = valid_results.sort_values(
            "Silhouette Score",
            ascending=False
        ).iloc[0]

        best_model_name = best_row["Model"]

        st.success(f"🏆 Best Clustering Model: {best_model_name}")

        # =================================================
        # VISUALIZATION (FIRST 2 FEATURES)
        # =================================================
        best_model = models[best_model_name]
        labels = best_model.fit_predict(X_scaled)

        fig, ax = plt.subplots()
        ax.scatter(
            X_scaled[:, 0],
            X_scaled[:, 1],
            c=labels,
            cmap="tab10"
        )
        ax.set_xlabel(num_cols[0])
        ax.set_ylabel(num_cols[1])
        ax.set_title("Cluster Visualization")

        st.pyplot(fig)

        st.session_state.best_clustering_model = best_model_name
