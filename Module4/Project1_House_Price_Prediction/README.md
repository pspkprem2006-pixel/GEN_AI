# Project 1 - House Price Prediction (Regression)

Predict a house's **price** (a number) with `LinearRegression` - the
canonical first ML model. Per Module 4 notes, section 12.

## Files
- `house_price_prediction.py` - the full program
- `actual_vs_predicted.png` - scatter chart (created when you run it)

## What it does
1. **Generates its own synthetic dataset** - 200 houses with `Area`,
   `Bedrooms`, `Age`, `Location` (City-Center / Suburb / Rural) using
   `random_state=42` (fully reproducible, no CSV needed). The price
   follows the notes' underlying law (section 4.7):
   `price = 3000*Area + 500000*Bedrooms - 25000*Age + 1200000*(Suburb) + 300000`
   plus noise.
2. **Encodes** the text `Location` column with `pd.get_dummies(drop_first=True)`.
3. **Splits** 80/20 with `random_state=42`.
4. **Trains** `LinearRegression` (the 4-line rhythm: choose, fit, predict,
   evaluate).
5. **Evaluates** with MAE, RMSE and R².
6. **Interprets** the coefficients - which feature matters most.
7. **Saves** an actual-vs-predicted scatter that hugs the diagonal.

## How to run
```bash
python house_price_prediction.py
```
Needs: `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

## Expected output
```
----- MODEL PERFORMANCE (on unseen test data) -----
MAE  (avg error)         : ~126,000
RMSE (penalizes big miss): ~160,000
R2   : ~0.98  -> explains ~98% of price variation

----- WHAT THE MODEL LEARNED (coefficients) -----
   Location_Suburb     : +1,194,987 per unit
   Bedrooms            : +514,262 per unit
   ...
```
Exact numbers vary slightly with the noise seed, but R² is always ~0.98
and the coefficients stay close to the true law (Area ~+3,000,
Bedrooms ~+500,000, Age ~-25,000).

## Challenges
1. Add a new feature (e.g. `GarageSpace`) to the data and the model -
   does R² improve?
2. Swap `LinearRegression` for `RandomForestRegressor` - compare R².
3. Find the prediction for a 2000 sq ft, 3-bedroom, 5-year-old Suburb
   house by hand (section 4.7), then compare it with the model's output.
