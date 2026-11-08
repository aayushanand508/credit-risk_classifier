"""
Credit Risk Classifier Lab - Streamlit app.

Loads five pre-trained scikit-learn pipelines (no retraining) and evaluates them
on a test CSV uploaded by the user (or on the bundled test_data.csv).
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import METRIC_ORDER, compute_metrics

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
TARGET = "credit_risk"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
CLASS_NAMES = ["Good risk (0)", "Bad risk (1)"]

st.set_page_config(page_title="Credit Risk Classifier Lab", layout="wide")


SEED = 7
NUMERIC = ["duration_months", "credit_amount", "installment_rate", "residence_since",
           "age", "existing_credits", "num_dependents"]


def _find_file(name):
    for p in (ROOT / name, ROOT / "data" / name, ROOT / "model" / name):
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner="Saved model files not found - training models from the dataset...")
def train_fallback():
    """Rebuild the five models with the same split, seed and tuned settings as train_models.py."""
    path = _find_file("german_credit_full.csv")
    if path is None:
        raise FileNotFoundError(
            "Neither model/*.joblib nor german_credit_full.csv was found in the repository.")
    df = pd.read_csv(path)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    categorical = [c for c in X.columns if c not in NUMERIC]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    estimators = {
        "Logistic Regression": LogisticRegression(C=0.1, max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=25, weights="distance"),
        "Naive Bayes": GaussianNB(var_smoothing=0.1),
        "Support Vector Machine": SVC(C=10, gamma=0.01, probability=True, random_state=SEED),
    }
    models = {}
    for name, est in estimators.items():
        prep = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ])
        models[name] = Pipeline([("prep", prep), ("clf", est)]).fit(X_train, y_train)
    return models


@st.cache_resource
def load_model(name):
    """Load the saved pipeline; if the file is missing or unreadable, use the rebuilt one."""
    path = MODEL_DIR / MODEL_FILES[name]
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            pass
    return train_fallback()[name]


@st.cache_data
def load_default_test():
    return pd.read_csv(_find_file("test_data.csv"))


def required_features():
    pipe = load_model("Logistic Regression")
    return list(pipe.named_steps["prep"].feature_names_in_)


def score(name, df, threshold):
    """Run a saved pipeline on df; return metrics, predictions, probabilities."""
    pipe = load_model(name)
    X = df[required_features()]
    proba = pipe.predict_proba(X)[:, 1]
    if threshold == 0.5:
        pred = pipe.predict(X)          # model's native decision rule (matches README)
    else:
        pred = (proba >= threshold).astype(int)
    return compute_metrics(df[TARGET], pred, proba), pred, proba


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Controls")
st.sidebar.caption("German Credit (Statlog) - predict whether an applicant is a bad credit risk.")

uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
st.sidebar.download_button(
    "Download sample test_data.csv",
    data=load_default_test().to_csv(index=False),
    file_name="test_data.csv",
    mime="text/csv",
)

model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES))
threshold = st.sidebar.slider(
    "Decision threshold for 'bad risk'", 0.10, 0.90, 0.50, 0.05,
    help="At 0.50 the model's native prediction rule is used, which reproduces the README metrics. Other values threshold the predicted probability of bad risk.")

# ---------------------------------------------------------------- data
st.title("Credit Risk Classifier Lab")
st.write("Pick a model in the sidebar and evaluate it on held-out applicants. "
         "Models are loaded from disk and are never retrained on your upload.")

if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = f"Uploaded file: {uploaded.name}"
else:
    data = load_default_test()
    source = "Bundled test_data.csv (200 held-out applicants). Upload your own file in the sidebar to override."

missing = [c for c in required_features() + [TARGET] if c not in data.columns]
if missing:
    st.error("The uploaded CSV is missing required columns: " + ", ".join(missing))
    st.stop()
if not set(data[TARGET].unique()) <= {0, 1} or data[TARGET].nunique() < 2:
    st.error(f"Column '{TARGET}' must contain both classes 0 and 1.")
    st.stop()

st.caption(source)
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(data))
c2.metric("Bad-risk share", f"{data[TARGET].mean():.1%}")
c3.metric("Selected model", model_name)

tab_eval, tab_compare, tab_data = st.tabs(
    ["Model evaluation", "Compare all models", "Data preview"])

# ---------------------------------------------------------------- tab 1
with tab_eval:
    metrics, pred, proba = score(model_name, data, threshold)

    st.subheader("Evaluation metrics")
    cols = st.columns(6)
    for col, key in zip(cols, METRIC_ORDER):
        col.metric(key, f"{metrics[key]:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(data[TARGET], pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("ROC curve")
        fpr, tpr, _ = roc_curve(data[TARGET], proba)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {metrics['AUC']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Classification report")
    report = classification_report(data[TARGET], pred, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

# ---------------------------------------------------------------- tab 2
with tab_compare:
    st.subheader("All five models on the same data")
    rows, curves = [], {}
    for name in MODEL_FILES:
        m, _, p = score(name, data, threshold)
        rows.append({"ML Model Name": name, **m})
        curves[name] = roc_curve(data[TARGET], p)[:2]
    table = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(table.style.highlight_max(axis=0, color="#cfe8cf").format("{:.4f}"),
                 width="stretch")
    st.caption("Green cell = best model for that metric.")

    st.bar_chart(table[["Accuracy", "AUC", "F1", "MCC"]])

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=8, loc="lower right")
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------- tab 3
with tab_data:
    st.subheader("First 50 rows of the evaluated data")
    st.dataframe(data.head(50), width="stretch")
    st.write("Required input columns:")
    st.code(", ".join(required_features() + [TARGET]))
