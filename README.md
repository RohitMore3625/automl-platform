# 🤖 AutoML Intelligence Platform

An end-to-end AutoML web app built with **Streamlit** and **scikit-learn**. Upload a dataset, pick a target column, and the app detects the problem type, trains and compares models, and explains the results with SHAP.

Made by **Rohit More**.

## ✨ Features

- **Data ingestion**: upload CSV or Excel files, with instant stats on rows, columns, missing values, and duplicates
- **Automatic problem detection**: Regression, Classification, Clustering, or Time Series based on your target column
- **Optimization preference**: choose Fast, Balanced, or High Accuracy to control how many models are tried
- **Regression**: Linear Regression, Random Forest, Gradient Boosting, compared on R², RMSE, and MAE
- **Classification**: Logistic Regression, Random Forest, Gradient Boosting, compared on Accuracy, Precision, Recall, and F1
- **Clustering**: KMeans and Agglomerative clustering, ranked by silhouette score, with a cluster plot
- **Time Series**: Naive forecast, Moving Average, and Linear Regression with lag features, using a time-aware train/test split
- **Model export**: download the best trained pipeline as a `.pkl` file
- **Explainability**: global and per-prediction SHAP explanations

## 🗂️ Project Structure

```
automl-platform/
├── home.py                  # Main page: upload, strategy, problem detection
├── requirements.txt
├── .gitignore
└── pages/
    ├── 2_Regression.py
    ├── 3_Classification.py
    ├── 4_Clustering.py
    ├── 5_TimeSeries.py
    └── 6_Explainability.py
```

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/RohitMore3625/automl-platform.git
cd automl-platform

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run home.py
```

The app opens at `http://localhost:8501`.

## 📖 How to Use

1. Open the **Home** page and upload a CSV or Excel file.
2. Choose an optimization preference (Fast, Balanced, or High Accuracy).
3. Select a target column. Leave it as "None" for clustering.
4. Click **Go to AutoML Training** to open the matching page.
5. Train and compare models, then download the best one.
6. Open **Explainability** to see SHAP plots (available after Regression or Classification).

## 🛠️ Tech Stack

Python · Streamlit · pandas · NumPy · scikit-learn · SHAP · Matplotlib · joblib

## 📌 Notes

- Time Series mode needs a column that is already parsed as a datetime type.
- Explainability works only for models trained on the Regression or Classification pages.

## 📄 License

This project is open for learning and personal use.
