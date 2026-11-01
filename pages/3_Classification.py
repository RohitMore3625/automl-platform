import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import joblib
import io

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Classification AutoML",
    page_icon="📊",
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
    "best_pipeline": None,
    "best_pipeline_problem_type": None,
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

if st.session_state.target is None:
    st.error("No target selected. Go back to Home page.")
    st.stop()

if st.session_state.problem_type != "Classification":
    st.error(f"Problem type is {st.session_state.problem_type}, not Classification.")
    st.stop()

# =====================================================
# LOAD STATE
# =====================================================
df = st.session_state.df.copy()
target = st.session_state.target
preference = st.session_state.preference

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<h1>📊 Classification AutoML</h1>
<p style="color:#9ca3af">Train, compare, and export classification models</p>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# DATA OVERVIEW
# =====================================================
with card():
    st.subheader("📂 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Target", target)

    with st.expander("Preview Data"):
        st.dataframe(df.head(), use_container_width=True)

# =====================================================
# SPLIT FEATURES & TARGET
# =====================================================
X = df.drop(columns=[target])
y = df[target]

# =====================================================
# PREPROCESSING
# =====================================================
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# =====================================================
# TRAIN TEST SPLIT
# =====================================================
# NOTE: stratify=y crashes if any class has fewer than 2 members
# (can't stratify a class you can't split). Fall back to a plain
# split in that case instead of letting the app crash.
value_counts = y.value_counts()
can_stratify = (value_counts >= 2).all() and len(value_counts) > 1

if can_stratify:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
else:
    st.warning(
        "Some classes have too few samples to stratify the split. "
        "Falling back to a random (non-stratified) split."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

# =====================================================
# MODEL ZOO
# =====================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

if preference == "Fast":
    models = {
        "Logistic Regression": models["Logistic Regression"]
    }

# =====================================================
# RUN AUTOML
# =====================================================
with card():
    st.subheader("🚀 Run AutoML")

    run = st.button("Train & Compare Models", use_container_width=True)

    if run:
        results = []

        with st.spinner("Training models..."):
            for name, model in models.items():

                pipeline = Pipeline([
                    ("prep", preprocessor),
                    ("model", model)
                ])

                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)

                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, preds),
                    "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
                    "F1": f1_score(y_test, preds, average="weighted", zero_division=0)
                })

        results_df = pd.DataFrame(results)

        # =================================================
        # SELECT BEST MODEL
        # =================================================
        best_row = results_df.sort_values("F1", ascending=False).iloc[0]
        best_model_name = best_row["Model"]

        st.success(f"🏆 Best Model: {best_model_name}")

        st.subheader("📊 Model Performance")
        st.dataframe(results_df, use_container_width=True)

        # =================================================
        # TRAIN BEST PIPELINE
        # =================================================
        best_pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", models[best_model_name])
        ])

        best_pipeline.fit(X_train, y_train)

        st.session_state.best_pipeline = best_pipeline
        st.session_state.best_pipeline_problem_type = "Classification"

# =====================================================
# EXPORT MODEL
# =====================================================
if (
    st.session_state.best_pipeline is not None
    and st.session_state.get("best_pipeline_problem_type") == "Classification"
):

    with card():
        st.subheader("📦 Export Model")

        buffer = io.BytesIO()
        joblib.dump(st.session_state.best_pipeline, buffer)
        buffer.seek(0)

        st.download_button(
            label="⬇️ Download Model (.pkl)",
            data=buffer,
            file_name="best_classification_model.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )
