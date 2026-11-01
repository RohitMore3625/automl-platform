import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Model Explainability",
    page_icon="🔍",
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
    "best_pipeline": None,
    "best_pipeline_problem_type": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================================
# VALIDATION
# =====================================================
if st.session_state.best_pipeline is None:
    st.error("No trained model found. Run Regression or Classification AutoML first.")
    st.stop()

if st.session_state.df is None or st.session_state.target is None:
    st.error("Dataset or target missing.")
    st.stop()

# =====================================================
# LOAD STATE
# =====================================================
pipeline = st.session_state.best_pipeline
df = st.session_state.df.copy()
target = st.session_state.target
problem_type = st.session_state.get("best_pipeline_problem_type")

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<h1>🔍 Model Explainability (SHAP)</h1>
<p style="color:#9ca3af">
Understand global and local model behavior
</p>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# PREPARE DATA
# =====================================================
if target not in df.columns:
    st.error(f"Target column '{target}' not found in the current dataset.")
    st.stop()

X = df.drop(columns=[target])

preprocessor = pipeline.named_steps["prep"]
model = pipeline.named_steps["model"]

X_processed = preprocessor.transform(X)

# NOTE: OneHotEncoder inside the ColumnTransformer returns a sparse
# matrix by default. SHAP explainers/plots expect dense numpy arrays,
# so this used to crash as soon as any categorical column existed.
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()
X_processed = np.asarray(X_processed)

# Cap sample size for SHAP — large datasets make TreeExplainer/summary_plot
# extremely slow.
MAX_SHAP_ROWS = 500
if X_processed.shape[0] > MAX_SHAP_ROWS:
    st.info(f"Using a random sample of {MAX_SHAP_ROWS} rows for SHAP (dataset is larger).")
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(X_processed.shape[0], MAX_SHAP_ROWS, replace=False)
    X_processed = X_processed[sample_idx]

# Feature names
try:
    feature_names = preprocessor.get_feature_names_out()
except Exception:
    feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

# =====================================================
# BUILD SHAP EXPLAINER
# =====================================================
with st.spinner("Computing SHAP values..."):
    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_processed)

        shap_values = explainer.shap_values(X_processed)
        expected_value = explainer.expected_value
    except Exception as e:
        st.error(f"Could not compute SHAP values for this model: {e}")
        st.stop()

# =====================================================
# NORMALIZE SHAP OUTPUT SHAPE
# =====================================================
# For classifiers, SHAP can return either:
#  - a list of arrays, one per class (older API)
#  - a single 3D array (n_samples, n_features, n_classes) (newer API)
# The original code always assumed a single 2D array, which crashed on
# any classifier. This picks a representative class: the positive class
# for binary classification, or the first class otherwise.
class_index = None

if isinstance(shap_values, list):
    class_index = 1 if len(shap_values) > 1 else 0
    shap_values_plot = shap_values[class_index]
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[class_index]

elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    class_index = 1 if shap_values.shape[-1] > 1 else 0
    shap_values_plot = shap_values[:, :, class_index]
    if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
        expected_value = np.asarray(expected_value)[class_index]

else:
    shap_values_plot = np.asarray(shap_values)
    if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
        expected_value = np.asarray(expected_value).ravel()[0]

if class_index is not None and problem_type == "Classification":
    classes_ = getattr(model, "classes_", None)
    label = classes_[class_index] if classes_ is not None and class_index < len(classes_) else class_index
    st.caption(f"Showing SHAP values for predicted class: **{label}**")

# =====================================================
# GLOBAL EXPLANATION
# =====================================================
with card():
    st.subheader("🌍 Global Feature Importance")

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values_plot,
        X_processed,
        feature_names=feature_names,
        show=False
    )
    st.pyplot(fig)
    plt.close(fig)

# =====================================================
# LOCAL EXPLANATION
# =====================================================
with card():
    st.subheader("🎯 Individual Prediction Explanation")

    max_idx = X_processed.shape[0] - 1
    row_idx = st.slider(
        "Select Row Index",
        min_value=0,
        max_value=max_idx,
        value=0
    )

    explanation = shap.Explanation(
        values=shap_values_plot[row_idx],
        base_values=expected_value,
        data=X_processed[row_idx],
        feature_names=feature_names
    )

    fig2, ax2 = plt.subplots()
    shap.waterfall_plot(explanation, show=False)
    st.pyplot(fig2)
    plt.close(fig2)
