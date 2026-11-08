"""
Train five classifiers on the German Credit (Statlog) dataset and save them.

Each model is stored as a full scikit-learn Pipeline (preprocessing + classifier),
so the Streamlit app can score a raw test CSV without retraining anything.

Run:  python train_models.py
Outputs:
    model/*.joblib        one saved pipeline per classifier
    test_data.csv         held-out test split (raw features + target column)
    model/metrics.csv     evaluation metrics of every model on the test split
"""
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import compute_metrics

warnings.filterwarnings("ignore")

SEED = 7
TARGET = "credit_risk"
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"

NUMERIC = ["duration_months", "credit_amount", "installment_rate", "residence_since",
           "age", "existing_credits", "num_dependents"]


def make_preprocessor(categorical):
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


def main():
    df = pd.read_csv(ROOT / "data" / "german_credit_full.csv")
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    categorical = [c for c in X.columns if c not in NUMERIC]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # Held-out test file used by the app (raw features + target)
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    # (estimator, hyper-parameter grid searched with 5-fold CV on the training split only)
    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"clf__C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"clf__max_depth": [3, 5, 7, 10], "clf__min_samples_leaf": [1, 5, 10]}),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 9, 15, 25], "clf__weights": ["uniform", "distance"]}),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]}),
        "Support Vector Machine": (
            SVC(probability=True, random_state=SEED),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.001]}),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, params = [], {}
    for name, (est, grid) in candidates.items():
        pipe = Pipeline([("prep", make_preprocessor(categorical)), ("clf", est)])
        search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        proba = best.predict_proba(X_test)[:, 1]
        pred = best.predict(X_test)
        m = compute_metrics(y_test, pred, proba)
        rows.append({"ML Model Name": name, **m})
        params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(best, MODEL_DIR / fname)
        print(f"{name:24s} {m}  best={params[name]}")

    pd.DataFrame(rows).to_csv(MODEL_DIR / "metrics.csv", index=False)
    (MODEL_DIR / "best_params.json").write_text(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
