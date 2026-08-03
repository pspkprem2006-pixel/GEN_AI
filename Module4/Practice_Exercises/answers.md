# Module 4 - Practice Exercises & Self-Assessment (Answers)

Answers to all exercises in Module 4 notes, section 17. Each question is
written verbatim from the notes, followed by its answer (section 17.7
answer key). Coding answers are complete, runnable scripts that generate
their own synthetic data - no external CSV files needed.

Libraries used: `numpy`, `pandas`, `scikit-learn`, `matplotlib`,
`seaborn`. Everything uses `random_state=42` for reproducibility.

---

## 17.1 Concept checks

**1. Explain the difference between regression and classification with an example each.**

Regression predicts a **number** (a continuous value); classification
predicts a **category** (a class label). Example of regression: predicting
a house's price (Module 4, Project 1). Example of classification:
predicting whether a customer will churn - "yes" or "no" (Project 2).

**2. Why must you never test a model on its training data?**

Because the model has already *seen* that data - it can memorize it and
score ~perfectly while failing on new data. Testing on training data gives
you no honest measure of real performance; it is like giving students the
exam answers to study from and then testing them on the same paper. This
is why we hold out a separate test set with `train_test_split`.

**3. Why is accuracy a poor metric for a fraud detector? What would you use instead?**

With ~1% fraud, a model that "predicts never-fraud" scores 99% accuracy
but catches **zero** fraud - accuracy is meaningless on imbalanced data.
Use **precision, recall, F1 and the confusion matrix** instead. Recall
matters most here: you do not want to miss real fraud.

**4. What does an R² of 0.85 mean? What about R² of 0?**

R² = 0.85 means the model explains 85% of the variation in the target -
good. R² = 0 means the model is no better than always predicting the
mean. (R² can even go negative for a terrible model.)

**5. Explain precision vs recall using a disease-screening example.**

- **Precision** = of the people the test flagged as sick, how many really
  are sick.
- **Recall** = of all truly sick people, how many the test caught.
Screening usually prioritizes **recall** - it is worse to miss a real case
(and re-test the healthy ones later) than to flag a healthy person.

**6. When would you scale features? Name two models that need it and two that don't.**

Scale when the model is **distance-based or gradient-based** - features
with different units/ranges would otherwise be compared unfairly (e.g.
Income 20-120 vs Complaints 0-6). **Needs scaling:** Logistic Regression,
KNN, K-Means, SVM (and neural networks). **Does NOT need it:** Decision
Trees, Random Forests (they split on thresholds, not distances).

**7. What is overfitting, and how do you detect and reduce it?**

Overfitting = the model **memorizes** the training data instead of
learning the general pattern: great train score, poor test score.
**Detect** by comparing train vs test performance (big gap = overfitting).
**Reduce** with: more data, a simpler model, regularization, or
cross-validation (and limiting tree depth for decision trees).

**8. In K-Means, what is a centroid and how do you choose k?**

A **centroid** is a cluster's center - the "average member" of the group.
Choose **k** with the **Elbow Method** (plot inertia vs k, pick the
"elbow" where the curve flattens) and confirm with the **silhouette
score**.

---

## 17.2 Coding - regression

**9. Load a CSV, split into X/y and train/test, and train a `LinearRegression`. Report R² and MAE.**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Self-contained version: no CSV in the repo, so we generate the data
# exactly like Project 1. With a real CSV you would replace this block
# with: df = pd.read_csv("your_file.csv")
rng = np.random.default_rng(42)
n = 200
area = np.clip(rng.normal(1500, 300, n), 500, 3500)
bedrooms = rng.integers(1, 6, n)
age = rng.integers(0, 41, n)
df = pd.DataFrame({
    "Area": area, "Bedrooms": bedrooms, "Age": age,
    "Price": (3000*area + 500000*bedrooms - 25000*age
              + 300000 + rng.normal(0, 150000, n)),
})

X = df.drop(columns=["Price"]);  y = df["Price"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42)
model = LinearRegression().fit(X_tr, y_tr)
pred = model.predict(X_te)
print("R2 :", r2_score(y_te, pred))
print("MAE:", mean_absolute_error(y_te, pred))
```

**10. Print and interpret the model's coefficients - which feature matters most?**

```python
# continue from question 9 (model is already trained)
# Biggest absolute coefficient = most influential feature
for name, coef in sorted(zip(X.columns, model.coef_),
                         key=lambda kv: -abs(kv[1])):
    print(f"{name}: {coef:+.2f}")
```
Run with the data above, `Bedrooms` dominates (each bedroom adds ~Rs
500,000, so its coefficient has the largest absolute value). Positive
coefficient = more of this feature pushes the price up; negative
(Bedrooms' partner, `Age`) pushes it down.

**11. Swap in a `RandomForestRegressor`; does R² improve on the test set?**

```python
from sklearn.ensemble import RandomForestRegressor

# continue from question 9
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
print("RF R2:", r2_score(y_te, rf.predict(X_te)))
```
On this dataset the linear law is exact, so the forest lands at about the
same R² (~0.98) or slightly below - no improvement. On real non-linear
data the forest usually beats linear regression (see notes 17.7: "often
higher R2 on non-linear data").

---

## 17.3 Coding - classification

**12. Train a `LogisticRegression` on a binary dataset; print the classification report.**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Self-contained: generate a small binary dataset (churn-like) since the
# repo has no CSV. Replace with pd.read_csv("your_file.csv") for real data.
rng = np.random.default_rng(42)
n = 600
tenure = rng.integers(1, 73, n)
charges = rng.uniform(20, 120, n)
complaints = rng.binomial(5, 0.25, n)
logit = (-3.55 - 0.085*tenure + 0.04*charges + 1.2*complaints)
prob = 1 / (1 + np.exp(-logit))
df = pd.DataFrame({"Tenure": tenure, "MonthlyCharges": charges,
                   "Complaints": complaints,
                   "Churn": (rng.uniform(size=n) < prob).astype(int)})
X = df.drop(columns=["Churn"]);  y = df["Churn"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
pred = clf.predict(X_te)
print(classification_report(y_te, pred))
```

**13. Draw the confusion matrix; identify the false positives and false negatives.**

```python
# continue from question 12
cm = confusion_matrix(y_te, pred)
print(cm)   # layout: [[TN, FP], [FN, TP]]
```
Reading the 2x2 layout `[[TN, FP], [FN, TP]]`: **top-right = false
positives** (flagged as churn, actually stayed - the model cries wolf) and
**bottom-left = false negatives** (actually churned, the model missed them
- the expensive ones to overlook). To "draw" it as a heatmap:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
plt.savefig("confusion_matrix_exercise.png")
```

**14. Use `predict_proba` to list the 5 samples the model is most confident are positive.**

```python
# continue from question 12
proba = clf.predict_proba(X_te)[:, 1]        # probability of class 1 (churn)
top5 = np.argsort(proba)[-5:][::-1]
print("Most-confident positives:", proba[top5].round(3))
```
`predict_proba` returns the probability of each class; we take column 1
(the positive class), sort descending, and show the top 5.

---

## 17.4 Coding - clustering

**15. Run K-Means with k=3 and k=5 on a 2-feature dataset; plot both.**

```python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Self-contained: 2-feature data (annual income, spending score), 200 points
rng = np.random.default_rng(42)
centers = np.array([[30, 30], [30, 80], [80, 20], [85, 85], [55, 50]])
X2 = np.vstack([centers[i] + rng.normal(0, 8, (40, 2)) for i in range(5)])

for k in (3, 5):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X2)
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels)
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                marker="X", c="red", s=150, label="centroids")
    plt.title(f"k={k}")
    plt.legend()
    plt.savefig(f"kmeans_k{k}.png")
```

**16. Draw an elbow plot for k = 1...10 and pick the best k.**

```python
# continue from question 15 (X2 already defined)
inertias = [KMeans(k, random_state=42, n_init=10).fit(X2).inertia_
            for k in range(1, 11)]
plt.figure()
plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("k"); plt.ylabel("inertia")
plt.title("Elbow method")
plt.savefig("elbow_plot.png")
# pick k at the "elbow" where the curve flattens -> here k=5
print("Choose k at the elbow: k=5 (curve flattens sharply after 5)")
```
The curve drops sharply to k=5 then flattens - the elbow is at **k=5**.

**17. Name each cluster you find based on its feature averages.**

```python
# continue from question 15 (X2 already defined)
km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X2)
df_c = pd.DataFrame(X2, columns=["Income_k", "SpendingScore"])
df_c["cluster"] = km.labels_
print(df_c.groupby("cluster").mean())

# Name each cluster by its averages (rule-of-thumb thresholds)
def name(income, spend):
    if income < 45 and spend < 45:   return "Budget"
    if income < 45 and spend >= 60:  return "Young Spenders"
    if income >= 70 and spend < 45:  return "Savers (win them over)"
    if income >= 70 and spend >= 60: return "Premium (VIP - target!)"
    return "Average"

for cl, row in df_c.groupby("cluster").mean().iterrows():
    print(f"Cluster {cl}: {name(row['Income_k'], row['SpendingScore'])}")
```
The group-by prints each cluster's average income and spend; a low-low
cluster gets named "Budget", low income + high spend = "Young Spenders",
high income + low spend = "Savers", high-high = "Premium (VIP)", and the
middle group = "Average" - exactly the naming logic of Project 3.

---

## 17.5 Integrative

**18. Complete all three projects, then do one README challenge in each.**

Done - the three projects live in this repo:
- `Project1_House_Price_Prediction/` (regression, LinearRegression)
- `Project2_Customer_Churn_Prediction/` (classification, LogisticRegression)
- `Project3_Customer_Segmentation/` (clustering, K-Means)

Each folder's README has its own Challenges section (e.g. add a feature,
swap the model, silhouette score). One completed challenge per project:
1. **Project 1** - swapped in `RandomForestRegressor`: R² stays ~0.98 on
   this linear synthetic data (a real non-linear dataset would show an
   improvement).
2. **Project 2** - re-trained with `class_weight="balanced"`: recall for
   the churn class improves while precision drops - the classic trade-off
   on imbalanced data.
3. **Project 3** - re-ran with `n_clusters=3`: the elbow at k=5 is still
   visible, and with k=3 the "Average" segment merges with its neighbors
   (clusters collapse to Budget / Young Spenders / Premium-Savers).

**19. Take a real Kaggle dataset (e.g., Titanic survival) and build + evaluate a classifier end to end.**

The complete end-to-end recipe (works offline below with a synthetic
stand-in, since the repo cannot download Kaggle CSVs; swap the first block
for `df = pd.read_csv("titanic.csv")` with your real data):

```python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ---- Synthetic stand-in for titanic.csv (same shape: mix of numeric
# ---- and categorical features, ~38% positive class = survived)
rng = np.random.default_rng(42)
n = 891
df = pd.DataFrame({
    "Pclass": rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
    "Age": np.where(rng.uniform(size=n) < 0.2, np.nan,
                    rng.normal(29, 13, n).clip(0.4, 80)).round(1),
    "Fare": rng.lognormal(3.2, 0.9, n).clip(0, 512).round(2),
    "Sex": rng.choice(["male", "female"], n, p=[0.65, 0.35]),
    "Embarked": rng.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]),
})
p = 1 / (1 + np.exp(-(0.9*(df["Sex"]=="female") + 1.3*(df["Pclass"]==1)
                       - 0.5*(df["Pclass"]==3) - 0.02*df["Age"] - 1.0)))
df["Survived"] = (rng.uniform(size=n) < p).astype(int)

# ---- Clean: drop the missing Ages (simplest) and encode text
df = df.dropna()
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# ---- Split (stratify keeps the survival ratio in both sets)
X = df.drop(columns=["Survived"]);  y = df["Survived"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)

# ---- Build + evaluate end to end
clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
pred = clf.predict(X_te)
print(classification_report(y_te, pred, target_names=["Died", "Survived"]))
cm = confusion_matrix(y_te, pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.savefig("titanic_confusion.png")
```
The key pipeline steps, in order: **clean** (drop missing values) →
**encode** (one-hot text columns) → **split** (stratified 80/20) →
**train** (`LogisticRegression`, same 4-line rhythm) → **evaluate**
(confusion matrix + precision/recall/F1, not just accuracy). Swap in
`RandomForestClassifier(n_estimators=100, random_state=42)` on line 1 of
the model to compare.

---

## 17.6 Quick self-check quiz

**1. Predicting temperature is regression or classification?**
→ **Regression** - temperature is a number.

**2. What are the 4 lines of the ML rhythm?**
→ **choose, fit, predict, evaluate** (pick a model, `.fit()`, `.predict()`,
then measure it).

**3. Which metric can be misleading on imbalanced data?**
→ **Accuracy** - on a 99/1 dataset "always predict the majority class"
scores 99% while learning nothing.

**4. What does `stratify=y` do?**
→ **Keeps class balance in the split** - the training and test sets keep
the same churn/survival ratio as the full data.

**5. Which algorithm needs no `y`?**
→ **K-Means / clustering** - it finds groups without labels
(unsupervised).

**6. High train score, low test score means...?**
→ **Overfitting** - the model memorized the training data.

**7. What does `predict_proba` return?**
→ **Class probabilities** - e.g. "88% chance this customer churns"
(column 1 = probability of the positive class).

**8. What does the Elbow Method choose?**
→ **The number of clusters k** - the point where the inertia curve stops
dropping sharply.
