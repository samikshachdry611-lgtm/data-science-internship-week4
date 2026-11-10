import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
file_name = 'Sample data (1).xlsx'
df = pd.read_excel(file_name)

print("=== 1. DATASET OVERVIEW ===")
print(f"Total Rows & Columns: {df.shape}")
print(df.head())

print("\n=== 2. SUMMARY STATISTICS ===")
print(df[['Units Sold', 'Sale Price', 'Gross Sales', 'Profit']].describe())

print("\n=== 3. TOTAL SALES & PROFIT BY COUNTRY ===")
country_perf = df.groupby('Country')[[' Sales', 'Profit']].sum().reset_index()
print(country_perf.sort_values(by=' Sales', ascending=False))

print("\n=== 4. TOTAL SALES & PROFIT BY PRODUCT ===")
product_perf = df.groupby('Product')[[' Sales', 'Profit']].sum().reset_index()
print(product_perf.sort_values(by=' Sales', ascending=False))

# -------------------------------------------------------------
# 5. REGRESSION MODEL TO PREDICT SALES
# -------------------------------------------------------------
print("\n=== 5. SALES PREDICTION REGRESSION MODEL ===")
# Define features (X) and target variable (y)
X = df[['Units Sold', 'Manufacturing Price', 'Sale Price', 'Gross Sales', 'COGS']]
y = df[' Sales']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model R-squared (Accuracy Score): {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")