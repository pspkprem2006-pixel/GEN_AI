"""Project 2 - Customer Churn Prediction (Supervised - Classification).

Predicts whether a customer WILL LEAVE (a CATEGORY: churn yes/no) with
LogisticRegression - evaluated properly with precision, recall, F1 and a
confusion matrix, because churn is imbalanced (~30%).

Per Module 4 notes, section 13:
- encode Contract with get_dummies + drop_first
- stratified train/test split (stratify keeps the churn ratio in both sets)
- StandardScaler + LogisticRegression in ONE Pipeline (avoids leakage)

The script GENERATES its own synthetic dataset (600 customers) with
random_state=42 so every run is reproducible. It saves a confusion-matrix
heatmap (PNG) and demonstrates predict_proba on a risky new customer.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)


def make_synthetic_customers(n=600, seed=42):
    """Generate the synthetic churn dataset described in the notes."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, n)
    monthly_charges = rng.uniform(20, 120, n).round(2)
    complaints = rng.binomial(5, 0.25, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n,
                          p=[0.45, 0.30, 0.25])

    # True underlying law: short tenure, high bills, many complaints and a
    # month-to-month contract all push the churn probability up.
    logit = (-3.55 - 0.085 * tenure + 0.04 * monthly_charges
             + 1.2 * complaints
             + 2.0 * (contract == "Month-to-month")
             + 0.8 * (contract == "One year"))
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.uniform(size=n) < prob

    return pd.DataFrame({
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Complaints": complaints,
        "Contract": contract,
        "Churn": churn.astype(int),
    })


def main():
    print("=" * 60)
    print("PROJECT 2 - CUSTOMER CHURN PREDICTION (CLASSIFICATION)")
    print("=" * 60)

    # 1. Prepare - generate data, encode, split (stratified)
    df = make_synthetic_customers()
    churn_rate = df["Churn"].mean()
    print(f"Generated synthetic dataset: {df.shape[0]} customers, "
          f"churn rate = {churn_rate:.0%} (imbalanced!)")

    df_encoded = pd.get_dummies(df, columns=["Contract"], drop_first=True)
    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]} "
          f"(churn ratio kept equal in both by stratify=y)")

    # 2. Train - scale + model in one Pipeline (avoids leakage)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 3. Evaluate - accuracy is NOT enough on imbalanced data
    print()
    print("----- MODEL PERFORMANCE (on unseen test data) -----")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    rep = classification_report(y_test, y_pred, target_names=["Stay", "Churn"])
    print(rep)

    # 4. Confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    print("----- CONFUSION MATRIX (rows=actual, cols=predicted) -----")
    print(f"[[ TN={cm[0,0]:3d}  FP={cm[0,1]:3d} ]")
    print(f" [ FN={cm[1,0]:3d}  TP={cm[1,1]:3d} ]]")
    print("top-right = false positives (flagged churn, actually stayed)")
    print("bottom-left = false negatives (actually churned, we missed them)")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stay", "Churn"],
                yticklabels=["Stay", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Churn Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=120)
    print()
    print("Saved chart: confusion_matrix.png")

    # 5. predict_proba - a confidence a business can act on
    # (Month-to-month is the dropped reference column after one-hot encoding,
    # so the risky new customer is a 0 in both contract dummy columns.)
    new = pd.Series(0.0, index=X.columns)
    new["Tenure"] = 3
    new["MonthlyCharges"] = 95.0
    new["Complaints"] = 4
    prob = model.predict_proba(pd.DataFrame([new]))[0, 1]
    verdict = "WILL CHURN" if prob >= 0.5 else "WILL STAY"
    print()
    print("----- PREDICTION FOR A NEW CUSTOMER -----")
    print("Customer profile: 3mo tenure, $95/mo, 4 complaints, "
          "month-to-month contract")
    print(f"  {verdict} (churn probability {prob * 100:.0f}%)")
    print("A business can act on this confidence - e.g. offer a "
          "retention discount.")


if __name__ == "__main__":
    main()
