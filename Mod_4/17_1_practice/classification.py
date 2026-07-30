import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
# Replace 'your_file.csv' with your actual CSV file path
df = pd.read_csv('your_file.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print('Classification Report:')
print(classification_report(y_test, pred))

print('\nConfusion Matrix:')
print(confusion_matrix(y_test, pred))

# Probability predictions
proba = model.predict_proba(X_test)[:, 1]

print('\nTop 5 most confident positive predictions:')
for i in proba.argsort()[-5:][::-1]:
    print(f'Sample {i}: probability = {proba[i]:.4f}')
