import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("Dataset/feature_engineered_data.csv")

# ==========================================
# Features and Target
# ==========================================

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================
# Function to Evaluate Models
# ==========================================

results = []

def evaluate_model(name, model):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    try:
        probabilities = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probabilities)
    except:
        roc_auc = 0

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ])

    print("\n" + "-" * 50)
    print(name)
    print("-" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    return model

# ==========================================
# Logistic Regression
# ==========================================

lr_model = evaluate_model(
    "Logistic Regression",
    LogisticRegression(max_iter=1000)
)

# ==========================================
# Random Forest
# ==========================================

rf_model = evaluate_model(
    "Random Forest",
    RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
)

# ==========================================
# XGBoost
# ==========================================

try:
    from xgboost import XGBClassifier

    xgb_model = evaluate_model(
        "XGBoost",
        XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )
    )

except Exception as e:
    print("\nXGBoost not available.")
    print("Reason:", e)

# ==========================================
# Results Table
# ==========================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]
)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)

# ==========================================
# Select Best Model
# ==========================================

best_model_name = results_df.sort_values(
    by="Accuracy",
    ascending=False
).iloc[0]["Model"]

print("\nBest Model:", best_model_name)

# ==========================================
# Save Random Forest
# ==========================================

joblib.dump(
    rf_model,
    "Model/loan_approval_model.pkl"
)

print("\nModel Saved:")
print("Model/loan_approval_model.pkl")

print("\nMODEL TRAINING COMPLETED")