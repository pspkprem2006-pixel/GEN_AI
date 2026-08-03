# Module 4 - Machine Learning Essentials

Completed hands-on tasks for Module 4 of the AI Powered Engineering
Upskilling Program.

This module covers the fundamentals of Machine Learning with scikit-learn:
the ML workflow, data preparation (encoding, scaling, train/test split),
regression, classification, clustering, and honest model evaluation.

All three syllabus projects are implemented. Each project **generates its
own synthetic dataset inside the script** (fixed `random_state=42`, so
results are reproducible) - no external CSV files needed - and saves a
PNG chart.

## Task Files (one file per task)

| Task | Folder / File | What it does |
|---|---|---|
| Project 1 | `Project1_House_Price_Prediction/house_price_prediction.py` | **Regression** - predicts house price with `LinearRegression` (encode Location, 80/20 split, MAE/RMSE/R², interpretable coefficients). Saves `actual_vs_predicted.png`. |
| Project 2 | `Project2_Customer_Churn_Prediction/churn_prediction.py` | **Classification** - predicts customer churn with `LogisticRegression` (stratified split, `StandardScaler`+model in a Pipeline, precision/recall/F1, confusion matrix, `predict_proba` on a risky new customer). Saves `confusion_matrix.png`. |
| Project 3 | `Project3_Customer_Segmentation/customer_segmentation.py` | **Clustering** - finds 5 customer segments with `KMeans` (elbow method, scaling, named business segments). Saves `segments_and_elbow.png`. |
| Practice | `Practice_Exercises/answers.md` | Answers to all of section 17: concept checks, runnable coding solutions (regression, classification, clustering), integrative tasks, and the self-check quiz. |

## The three ML paradigms, covered by the three projects

| Paradigm | Project | Model | Label |
|---|---|---|---|
| Supervised - Regression | 1 - House Price | `LinearRegression` | Price (a number) |
| Supervised - Classification | 2 - Churn | `LogisticRegression` | Churn (a category) |
| Unsupervised - Clustering | 3 - Segmentation | `KMeans` | None (no labels!) |

## How to run

Each project is standalone; run it from its own folder:

```bash
cd Project1_House_Price_Prediction
python house_price_prediction.py
```

```bash
cd Project2_Customer_Churn_Prediction
python churn_prediction.py
```

```bash
cd Project3_Customer_Segmentation
python customer_segmentation.py
```

No installs needed - the required libraries (`numpy`, `pandas`,
`scikit-learn`, `matplotlib`, `seaborn`, `scipy`) are already available.

## Expected results

- **Project 1:** R² ~0.98, MAE ~126k, RMSE ~161k; coefficients close to
  the true law (Area ~+3,000, Bedrooms ~+500,000, Age ~-25,000).
- **Project 2:** accuracy ~0.87, churn precision/recall ~0.81; the risky
  new customer (3mo tenure, $95/mo, 4 complaints, month-to-month) is
  flagged WILL CHURN with high confidence.
- **Project 3:** elbow at k=5; five named segments of 40 customers each
  (Budget, Young Spenders, Average, Savers, Premium) - matching the
  notes' sample output.

## Key lessons demonstrated

- The **4-line ML rhythm**: choose, fit, predict, evaluate.
- **Never test on training data** - always hold out a test set.
- **Accuracy alone is not enough** on imbalanced data - report
  precision/recall/F1 and the confusion matrix.
- **Scale before distance/gradient models** - use a Pipeline to avoid
  leakage.
- **`random_state=42` everywhere** - reproducible results.
- K-Means discovers structure with **no labels** - then we name the
  segments with business meaning.
