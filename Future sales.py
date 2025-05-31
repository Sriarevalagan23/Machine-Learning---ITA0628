import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample Sales Data (monthly sales for a store)
data = {
    'Year': [2020]*12 + [2021]*12 + [2022]*12,
    'Month': list(range(1,13))*3,
    'Sales': [
        200, 220, 215, 240, 260, 270, 300, 310, 305, 320, 330, 340,  # 2020
        350, 360, 355, 380, 390, 400, 420, 430, 425, 440, 450, 460,  # 2021
        470, 480, 475, 500, 520, 530, 550, 560, 555, 570, 580, 590   # 2022
    ]
}

df = pd.DataFrame(data)

# Feature Engineering: Create 'Time Index' to capture order of months
df['Time_Index'] = np.arange(len(df))

# Define features and target
X = df[['Year', 'Month', 'Time_Index']]
y = df['Sales']

# Split data (train on first 30 months, test on last 6)
X_train, X_test = X.iloc[:-6], X.iloc[-6:]
y_train, y_test = y.iloc[:-6], y.iloc[-6:]

# Initialize and train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict sales for test period
y_pred = model.predict(X_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot actual vs predicted sales
plt.figure(figsize=(10,5))
plt.plot(df['Time_Index'], y, label='Actual Sales', marker='o')
plt.plot(X_test['Time_Index'], y_pred, label='Predicted Sales', marker='x')
plt.xlabel('Time Index (Months)')
plt.ylabel('Sales')
plt.title('Future Sales Prediction')
plt.legend()
plt.show()

# Predict sales for future months (e.g., next 6 months)
future_months = pd.DataFrame({
    'Year': [2023]*6,
    'Month': list(range(1,7)),
    'Time_Index': np.arange(len(df), len(df)+6)
})

future_sales_pred = model.predict(future_months)
print("\nPredicted Sales for next 6 months (2023):", future_sales_pred)
