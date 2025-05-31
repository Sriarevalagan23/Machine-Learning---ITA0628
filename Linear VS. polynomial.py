import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Generate synthetic non-linear data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2.5, 4.4, 6.1, 8.6, 11.3, 14.9, 19.2, 24.3, 30.1, 36.6])  # non-linear relationship

# Linear Regression model
lin_model = LinearRegression()
lin_model.fit(X, y)
y_pred_lin = lin_model.predict(X)

# Polynomial Regression (degree 2)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

# R² scores
r2_linear = r2_score(y, y_pred_lin)
r2_poly = r2_score(y, y_pred_poly)

# Print model performance
print(f"Linear Regression R² Score: {round(r2_linear, 4)}")
print(f"Polynomial Regression (degree=2) R² Score: {round(r2_poly, 4)}")

# Plot results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred_lin, color='red', label='Linear Regression')
plt.plot(X, y_pred_poly, color='green', label='Polynomial Regression (degree=2)')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Comparison: Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)
plt.show()
