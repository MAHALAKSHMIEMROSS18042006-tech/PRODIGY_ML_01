# ==========================================
# Task 1: House Price Prediction using Linear Regression
# ==========================================

print("STARTING TASK 1...")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("train.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)

# Select features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

df = df[features + [target]]
df = df.dropna()

# Define X and y
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== Model Performance =====")
print("R2 Score:", round(r2, 4))
print("RMSE:", round(rmse, 2))

# Coefficients
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})

print("\n===== Feature Importance =====")
print(coefficients)

print("\nTask 1 Completed Successfully ✅")