"""Project 1 - House Price Prediction (Supervised - Regression).

Predicts the price of a house (a NUMBER) with LinearRegression.
Per Module 4 notes, section 12 (and 4.4 / 4.7):
    price = 3000*Area + 500000*Bedrooms - 25000*Age + 1200000*(Suburb) + 300000

The script GENERATES its own synthetic dataset (200 houses) with
random_state=42 so every run is reproducible. It then runs the full
pipeline: encode text -> split -> train -> evaluate -> interpret,
and saves an "actual vs predicted" scatter chart (PNG).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def make_synthetic_houses(n=200, seed=42):
    """Generate the synthetic house-price dataset described in the notes."""
    rng = np.random.default_rng(seed)

    area = np.clip(rng.normal(1500, 300, n), 500, 3500).round()
    bedrooms = rng.integers(1, 6, n)
    age = rng.integers(0, 41, n)
    location = rng.choice(["City-Center", "Suburb", "Rural"], n,
                          p=[0.4, 0.35, 0.25])

    # True underlying law (Module 4 notes, section 4.7) plus noise.
    base = 3000 * area + 500000 * bedrooms - 25000 * age + 300000
    suburb_bonus = np.where(location == "Suburb", 1200000, 0)
    noise = rng.normal(0, 150000, n)
    price = base + suburb_bonus + noise

    return pd.DataFrame({
        "Area": area.astype(int),
        "Bedrooms": bedrooms,
        "Age": age,
        "Location": location,
        "Price": price.round(),
    })


def main():
    print("=" * 60)
    print("PROJECT 1 - HOUSE PRICE PREDICTION (REGRESSION)")
    print("=" * 60)

    # 1. Prepare - features X, label y, encode text, split
    df = make_synthetic_houses()
    print(f"Generated synthetic dataset: {df.shape[0]} houses, "
          f"features: {list(df.columns)}")

    df_encoded = pd.get_dummies(df, columns=["Location"], drop_first=True)
    X = df_encoded.drop(columns=["Price"])
    y = df_encoded["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")

    # 2. Train (the 4-line rhythm: choose, fit, predict, evaluate)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 3. Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print()
    print("----- MODEL PERFORMANCE (on unseen test data) -----")
    print(f"MAE  (avg error)         : {mae:,.0f}")
    print(f"RMSE (penalizes big miss): {rmse:,.0f}")
    print(f"R2   : {r2:.3f}  -> explains {r2 * 100:.1f}% of price variation")

    # 4. Interpret the coefficients (which feature matters most?)
    print()
    print("----- WHAT THE MODEL LEARNED (coefficients) -----")
    coefs = sorted(zip(X.columns, model.coef_),
                   key=lambda kv: -abs(kv[1]))
    for name, coef in coefs:
        print(f"   {name:20s}: {coef:+,.0f} per unit")
    print(f"   {'(intercept)':20s}: {model.intercept_:+,.0f}")
    print()
    print("Interpretation: each sq ft adds ~Rs 3,000; each extra bedroom "
          "adds ~Rs 5,00,000;")
    print("each year of age subtracts ~Rs 24,000 - this is the power of "
          "linear regression:")
    print("it does not just predict, it explains.")

    # 5. Save the "actual vs predicted" scatter chart
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect fit (diagonal)")
    plt.xlabel("Actual price (Rs)")
    plt.ylabel("Predicted price (Rs)")
    plt.title("House Price: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=120)
    print()
    print("Saved chart: actual_vs_predicted.png (points hug the diagonal, "
          "R2 = %.3f)" % r2)


if __name__ == "__main__":
    main()
