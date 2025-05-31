# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Step 2: Sample Data (replace with your dataset or load CSV)
data = {
    'Brand': ['Apple', 'Samsung', 'Apple', 'OnePlus', 'Samsung', 'OnePlus', 'Apple'],
    'RAM_GB': [4, 6, 3, 8, 4, 12, 2],
    'Storage_GB': [64, 128, 32, 256, 64, 256, 32],
    'Battery_mAh': [3000, 4000, 2800, 4500, 3500, 4600, 2700],
    'Price': [700, 650, 600, 750, 620, 800, 580]  # Price in USD (target)
}

df = pd.DataFrame(data)

# Step 3: Encode categorical feature (Brand)
le = LabelEncoder()
df['Brand'] = le.fit_transform(df['Brand'])

# Step 4: Define features and target
X = df[['Brand', 'RAM_GB', 'Storage_GB', 'Battery_mAh']]
y = df['Price']

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Predict on test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print("Predicted Prices:", y_pred)
print("Actual Prices:", y_test.values)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
