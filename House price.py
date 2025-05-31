# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Sample Dataset (You can replace this with a CSV file)
data = {
    'Area': [1200, 1500, 800, 950, 1300, 1800],
    'Bedrooms': [3, 4, 2, 2, 3, 5],
    'Bathrooms': [2, 3, 1, 2, 2, 4],
    'Location': ['Urban', 'Urban', 'Rural', 'Suburban', 'Suburban', 'Urban'],
    'Price': [300000, 400000, 150000, 200000, 320000, 500000]
}

df = pd.DataFrame(data)

# Step 3: Encode Categorical Variable (Location)
label_encoder = LabelEncoder()
df['Location'] = label_encoder.fit_transform(df['Location'])

# Step 4: Define Features and Target
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Location']]
y = df['Price']

# Step 5: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict and Evaluate
y_pred = model.predict(X_test)

print("Predicted House Prices:", y_pred)
print("Actual House Prices:", y_test.values)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
