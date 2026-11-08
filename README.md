# Credit Risk Classifier Lab

Classification of loan applicants as good or bad credit risks using five machine learning models, with an interactive Streamlit app for evaluating the saved models on test data.

## a. Problem Statement

Lenders must decide whether an applicant is likely to repay a loan. A wrong approval of a risky applicant causes a financial loss, while a wrong rejection of a good applicant loses business. The goal of this project is to build and compare five classification models that predict whether an applicant is a bad credit risk (class 1) or a good credit risk (class 0) from the applicant's financial and personal attributes, and to deploy the trained models in an interactive web application for evaluation on unseen data.

## b. Dataset Description 

- Name: Statlog (German Credit Data)
- Source: UCI Machine Learning Repository (Prof. Hans Hofmann, University of Hamburg)
- Task: binary classification
- Instances: 1000 (700 good risk, 300 bad risk)
- Features: 20 input features (7 numeric, 13 categorical) plus the target column `credit_risk`
- Missing values: none
- Target encoding: `credit_risk` = 1 for bad credit risk (positive class), 0 for good credit risk. The original file codes these as 2 and 1 respectively.

Features: checking_status, duration_months, credit_history, purpose, credit_amount, savings_status, employment_since, installment_rate, personal_status, other_debtors, residence_since, property, age, other_installment_plans, housing, existing_credits, job, num_dependents, telephone, foreign_worker.

Preprocessing (inside each saved scikit-learn Pipeline, so raw CSV rows can be scored directly):
- Numeric features: standardised with StandardScaler
- Categorical features: one-hot encoded (unknown categories ignored)

Train/test split: 80/20 stratified split with random_state 7. 800 rows are used for training and hyper-parameter tuning (5-fold cross validation, scoring by ROC AUC); the remaining 200 rows (60 bad risk, 140 good risk) are saved as `test_data.csv` and used for every metric reported below.

## c. GitHub Repository Link [1 mark]

https://github.com/<YOUR-GITHUB-USERNAME>/credit-risk-classifier

Live Streamlit app: https://<YOUR-APP-NAME>.streamlit.app

Repository structure:

```
credit-risk-classifier/
|-- app.py                  Streamlit application
|-- evaluation.py           shared metric function
|-- train_models.py         trains, tunes and saves all five models
|-- requirements.txt
|-- README.md
|-- test_data.csv           held-out test split (raw features + target)
|-- data/
|   `-- german_credit_full.csv
`-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- support_vector_machine.joblib
    |-- metrics.csv
    `-- best_params.json
```

## d. Models Used [5 marks]

All five models were trained on the same training split. The positive class for Precision, Recall, F1 and MCC is bad credit risk (class 1). AUC is computed from predicted probabilities.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8200 | 0.8371 | 0.8000 | 0.5333 | 0.6400 | 0.5455 |
| Decision Tree | 0.7550 | 0.7664 | 0.6667 | 0.3667 | 0.4731 | 0.3557 |
| KNN | 0.7500 | 0.8171 | 0.8571 | 0.2000 | 0.3243 | 0.3336 |
| Naive Bayes | 0.8100 | 0.8445 | 0.7200 | 0.6000 | 0.6545 | 0.5292 |
| Support Vector Machine | 0.8000 | 0.8268 | 0.7273 | 0.5333 | 0.6154 | 0.4952 |

Hyper-parameters selected by grid search (5-fold CV on the training split):
- Logistic Regression: C = 0.1
- Decision Tree: max_depth = 5, min_samples_leaf = 10
- KNN: n_neighbors = 25, weights = distance
- Naive Bayes (Gaussian): var_smoothing = 0.1
- Support Vector Machine (RBF kernel): C = 10, gamma = 0.01

### Observations on model performance [3 marks]

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest accuracy (0.82) and highest MCC (0.5455) of all models, with strong precision (0.80). The regularised linear model handles the one-hot and scaled features well and is stable on this small dataset. It still misses almost half of the bad-risk applicants (recall 0.53). |
| Decision Tree | Lowest AUC of all models (0.7664); accuracy (0.755) and MCC (0.3557) are only marginally above KNN, and recall is only 0.37. Even with depth limited to 5 and a minimum leaf size of 10, a single tree is high-variance on 800 training rows and ranks applicants less reliably than the other models. |
| KNN | Highest precision (0.8571) but the lowest recall (0.20) and F1 (0.3243). It predicts bad risk only when the neighbourhood is strongly bad, so it is dominated by the 70 percent majority class. Its AUC (0.8171) is reasonable, which shows the ranking is good but the 0.5 cut-off is poorly suited to it. |
| Naive Bayes | Best AUC (0.8445), best recall (0.60) and best F1 (0.6545), with accuracy of 0.81. Despite the independence assumption it works well here and catches the most bad-risk applicants, at the cost of lower precision (0.72). Useful when missing a risky applicant is the costlier mistake. |
| Support Vector Machine | Balanced and consistent: accuracy 0.80, AUC 0.8268, MCC 0.4952. It is slightly behind Logistic Regression and Naive Bayes on every metric, so the non-linear RBF kernel gives no gain over the simpler models on this dataset. |
| Overall, Winner for your dataset? | Logistic Regression, because it has the highest accuracy and MCC (the most balanced single measure on an imbalanced target), high precision, and an AUC within 0.01 of the best. Naive Bayes is a close second and is the better choice if recall on bad-risk applicants matters most. Note that the test set has only 200 rows, so differences of 1 to 2 percentage points correspond to a few applicants and should be read cautiously. |

## Streamlit App Features

- Dataset upload option: upload a CSV containing all 20 input features and the `credit_risk` column (a sample file can be downloaded from the sidebar; if nothing is uploaded, the bundled `test_data.csv` is used)
- Model selection dropdown: choose any of the five saved models
- Display of evaluation metrics: Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix, ROC curve and classification report for the selected model
- Comparison tab that scores all five models on the same data, highlights the best value per metric, and plots all ROC curves
- Optional decision-threshold slider (the default of 0.50 reproduces the table above)

The app loads the saved models from the `model/` folder and never retrains them on uploaded data.

## How to Run Locally

```
pip install -r requirements.txt
python train_models.py      # optional: regenerates model/ and test_data.csv
streamlit run app.py
```

## Deployment on Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click New app, choose the repository and the main branch, select `app.py`, and click Deploy.
