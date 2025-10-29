# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load dataset
data = pd.read_csv("Salary_dataset.csv")

# Separate features (X) and target (y)
X = data[['YearsExperience']]   # independent variable
y = data['Salary']              # dependent variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                                    
#Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
                                                    
# Make predictions on test data
y_pred = model.predict(X_test)

# Compare predictions
# Print actual vs predicted
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': np.round(y_pred, 2)})
print(comparison)
# Evaluate performance
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("R² Score:", round(r2_score(y_test, y_pred), 2))

# Visualize the regression line
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regression Line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience (Linear Regression)')
plt.legend()
plt.show()

