# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create or Load Dataset
# You can replace this with your own CSV: df = pd.read_csv('car_data.csv')
data = {
    'Brand': ['Toyota', 'Honda', 'BMW', 'Toyota', 'Honda', 'BMW'],
    'Year': [2015, 2016, 2018, 2014, 2017, 2019],
    'Mileage': [50000, 40000, 30000, 60000, 35000, 25000],
    'Fuel_Type': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel'],
    'Price': [500000, 550000, 800000, 450000, 580000, 850000]
}

df = pd.DataFrame(data)

# Step 3: Encode Categorical Features
le_brand = LabelEncoder()
le_fuel = LabelEncoder()
df['Brand'] = le_brand.fit_transform(df['Brand'])
df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])

# Step 4: Define Features and Target
X = df[['Brand', 'Year', 'Mileage', 'Fuel_Type']]
y = df['Price']

# Step 5: Split into Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict and Evaluate
y_pred = model.predict(X_test)

print("Predicted Prices:", y_pred)
print("Actual Prices:", y_test.values)

# Step 8: Evaluation Metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
