import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (replace with your actual data or load from CSV)
data = {
    'Credit_Score': [700, 650, 600, 580, 720, 690, 710, 630, 590, 680],
    'Income': [50000, 42000, 38000, 36000, 54000, 48000, 52000, 41000, 39000, 47000],
    'Loan_Amount': [200000, 180000, 150000, 130000, 210000, 195000, 205000, 175000, 160000, 190000],
    'Employment_Status': ['Employed', 'Self-employed', 'Employed', 'Unemployed', 'Employed', 'Employed', 'Self-employed', 'Unemployed', 'Employed', 'Self-employed'],
    'Loan_Approved': ['Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Encode categorical variable
le = LabelEncoder()
df['Employment_Status'] = le.fit_transform(df['Employment_Status'])
df['Loan_Approved'] = le.fit_transform(df['Loan_Approved'])  # Yes=1, No=0

# Define features and target
X = df.drop('Loan_Approved', axis=1)
y = df['Loan_Approved']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
