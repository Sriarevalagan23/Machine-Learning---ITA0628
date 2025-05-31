import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Generate Sample Dataset
data = {
    'Age': [25, 45, 35, 50, 23, 40, 60, 30, 28, 48],
    'Income': [50000, 80000, 60000, 120000, 35000, 75000, 100000, 65000, 58000, 90000],
    'Loan_Amount': [20000, 50000, 25000, 30000, 10000, 40000, 60000, 22000, 18000, 45000],
    'Loan_Term': [12, 24, 18, 36, 12, 30, 48, 20, 15, 36],
    'Credit_History': [1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    'Credit_Score': ['Good', 'Good', 'Average', 'Good', 'Poor', 'Average', 'Good', 'Average', 'Poor', 'Good']
}

df = pd.DataFrame(data)

# Step 2: Encode Target Variable
le = LabelEncoder()
df['Credit_Score'] = le.fit_transform(df['Credit_Score'])  # Good=1, Average=0, Poor=2 (or similar)

# Step 3: Split Features and Target
X = df.drop('Credit_Score', axis=1)
y = df['Credit_Score']

# Step 4: Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 6: Train Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluation
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
