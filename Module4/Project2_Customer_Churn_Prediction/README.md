# Project 2 - Customer Churn Prediction (Classification)

Predict whether a customer will **leave** (a category: churn yes/no) with
`LogisticRegression` - evaluated *properly* with precision, recall and F1,
because churn is imbalanced (~36%). Per Module 4 notes, section 13.

## Files
- `churn_prediction.py` - the full program
- `confusion_matrix.png` - heatmap (created when you run it)

## What it does
1. **Generates its own synthetic dataset** - 600 customers with `Tenure`,
   `MonthlyCharges`, `Complaints` and `Contract` (Month-to-month / One year
   / Two year) using `random_state=42` (fully reproducible, no CSV needed).
   The churn label comes from a logistic law: short tenure, high bills,
   many complaints and a month-to-month contract all raise churn risk.
2. **Encodes** `Contract` with `pd.get_dummies(drop_first=True)`.
3. **Splits** 80/20 with `stratify=y` - keeps the churn ratio in both sets.
4. **Trains** `StandardScaler` + `LogisticRegression(max_iter=1000)` in one
   **Pipeline** (avoids data leakage).
5. **Evaluates** with accuracy, precision, recall, F1 and a confusion
   matrix - accuracy alone is not enough on imbalanced data.
6. **Saves** a confusion-matrix heatmap.
7. **Predicts** a risky new customer (3mo tenure, $95/mo, 4 complaints,
   month-to-month) with `predict_proba` - a confidence a business can
   act on.

## How to run
```bash
python churn_prediction.py
```
Needs: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

## Expected output
```
Generated synthetic dataset: 600 customers, churn rate = 36% (imbalanced!)
Accuracy : ~0.87
      precision    recall  f1-score
Stay      0.90       0.90      0.90
Churn     0.81       0.81      0.81
...
PREDICTION FOR A NEW CUSTOMER:
  WILL CHURN (churn probability ~99-100%)
```
Accuracy ~0.87 closely matches the notes' sample (0.869). The exact
probability for the risky profile depends on the dataset - the module
notes' sample shows 88%; ours prints ~100%. Both make the same business
point: this customer is high-risk and should get a retention offer.

## Challenges
1. Swap `LogisticRegression` for `RandomForestClassifier` - does the
   churn recall improve?
2. Re-train with `class_weight="balanced"` - watch precision vs recall
   trade-off.
3. Pick a *different* synthetic customer (e.g. 5-year tenure, $50/mo,
   no complaints, two-year contract) and explain why the model predicts
   WILL STAY.
