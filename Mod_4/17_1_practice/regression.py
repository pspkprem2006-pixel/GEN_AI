import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
# Replace 'your_file.csv' with your actual CSV file path
# Example: df = pd.read_csv('house_prices.csv')
df = pd.read_csv('your_file.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
pred_lin = lin_reg.predict(X_test)

print('LinearRegression:')
print('R²:', r2_score(y_test, pred_lin))
print('MAE:', mean_absolute_error(y_test, pred_lin))
print('Coefficients:')
for feature, coef in zip(X.columns, lin_reg.coef_):
    print(f'{feature}: {coef}')

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg.fit(X_train, y_train)
pred_rf = rf_reg.predict(X_test)

print('\nRandomForestRegressor:')
print('R²:', r2_score(y_test, pred_rf))
print('MAE:', mean_absolute_error(y_test, pred_rf))
