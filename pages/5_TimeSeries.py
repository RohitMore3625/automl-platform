import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Time Series AutoML",
    page_icon="⏳",
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
    "best_time_series_model": None,
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

if st.session_state.problem_type != "Time Series":
    st.error(f"Problem type is {st.session_state.problem_type}, not Time Series.")
    st.stop()

# =====================================================
# LOAD DATA
# =====================================================
df = st.session_state.df.copy()
target = st.session_state.target

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<h1>⏳ Time Series AutoML</h1>
<p style="color:#9ca3af">Automated forecasting with time-aware validation</p>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# DATETIME COLUMN DETECTION
# =====================================================
datetime_cols = df.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns.tolist()

if len(datetime_cols) == 0:
    st.error("No datetime column detected. Time Series requires a datetime column.")
    st.stop()

date_col = datetime_cols[0]

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col)
df.set_index(date_col, inplace=True)

# =====================================================
# MINIMUM LENGTH CHECK
# =====================================================
# NOTE: with too few rows, the lagged-features model has nothing to
# train on and everything downstream (split, lag creation) breaks.
n_lags = 3
min_rows_needed = n_lags + 5  # a little slack for a meaningful train/test split

if len(df) < min_rows_needed:
    st.error(
        f"Not enough rows for time series forecasting. "
        f"Need at least {min_rows_needed} rows, found {len(df)}."
    )
    st.stop()

# =====================================================
# DATA OVERVIEW
# =====================================================
with card():
    st.subheader("📂 Time Series Overview")
    c1, c2 = st.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Target", target)

    with st.expander("Preview Data"):
        st.dataframe(df.head(), use_container_width=True)

# =====================================================
# TRAIN / TEST SPLIT (TIME AWARE)
# =====================================================
split_ratio = 0.8
split_idx = int(len(df) * split_ratio)
split_idx = max(split_idx, n_lags + 1)  # ensure the lag model has train rows

train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

y_train = train[target]
y_test = test[target]

# =====================================================
# MODEL 1 — NAIVE FORECAST
# =====================================================
naive_pred = np.repeat(y_train.iloc[-1], len(y_test))

# =====================================================
# MODEL 2 — MOVING AVERAGE
# =====================================================
window = min(5, len(y_train))
ma_value = y_train.rolling(window).mean().iloc[-1]
ma_pred = np.repeat(ma_value, len(y_test))

# =====================================================
# MODEL 3 — LINEAR REGRESSION WITH LAGS
# =====================================================
def create_lags(series, lags=3):
    data = pd.DataFrame({"y": series})
    for i in range(1, lags + 1):
        data[f"lag_{i}"] = data["y"].shift(i)
    return data.dropna()

lagged = create_lags(df[target], lags=n_lags)

X = lagged.drop(columns=["y"])
y = lagged["y"]

# `lagged` has `n_lags` fewer rows than `df` (the first n_lags rows are
# dropped since they don't have full lag history). Its row i corresponds
# to df row (i + n_lags), so to match the same time-based split point
# (split_idx in df's terms) we offset by n_lags rather than assuming a
# fixed "-3" as in the original code.
lag_split_idx = split_idx - n_lags

X_train_lr = X.iloc[:lag_split_idx]
X_test_lr = X.iloc[lag_split_idx:]
y_train_lr = y.iloc[:lag_split_idx]
y_test_lr = y.iloc[lag_split_idx:]

if len(X_train_lr) == 0 or len(X_test_lr) == 0:
    st.error("Not enough data after lag creation to train/test the lag model.")
    st.stop()

lr = LinearRegression()
lr.fit(X_train_lr, y_train_lr)
lr_pred = lr.predict(X_test_lr)

# =====================================================
# METRICS
# =====================================================
# NOTE: mean_squared_error's `squared=` argument was removed in current
# scikit-learn versions. np.sqrt() works on every version.
results = pd.DataFrame([
    {
        "Model": "Naive Forecast",
        "MAE": mean_absolute_error(y_test, naive_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, naive_pred))
    },
    {
        "Model": "Moving Average",
        "MAE": mean_absolute_error(y_test, ma_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, ma_pred))
    },
    {
        "Model": "Linear Regression (lags)",
        "MAE": mean_absolute_error(y_test_lr, lr_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test_lr, lr_pred))
    }
])

# =====================================================
# SELECT BEST MODEL
# =====================================================
best_model = results.sort_values("RMSE").iloc[0]["Model"]

with card():
    st.subheader("🏆 Best Forecasting Model")
    st.success(best_model)

    st.subheader("📊 Model Comparison")
    st.dataframe(results, use_container_width=True)

# =====================================================
# FORECAST VISUALIZATION
# =====================================================
with card():
    st.subheader("📈 Forecast Visualization")

    fig, ax = plt.subplots()
    ax.plot(train.index, y_train, label="Train")
    ax.plot(test.index, y_test, label="Actual")

    if best_model == "Naive Forecast":
        ax.plot(test.index, naive_pred, label="Forecast")
    elif best_model == "Moving Average":
        ax.plot(test.index, ma_pred, label="Forecast")
    else:
        ax.plot(y_test_lr.index, lr_pred, label="Forecast")

    ax.legend()
    ax.set_title("Time Series Forecast")

    st.pyplot(fig)

st.session_state.best_time_series_model = best_model
