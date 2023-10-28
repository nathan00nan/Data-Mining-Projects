import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv(r"C:\Users\pumpk\Desktop\Nathan's File\DOCUMENTS\My Notes\Projects\Passive Aggressive Regressor.csv", encoding='latin1')

data = data.dropna()

x = data[['Likes', 'Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
y = data['Impressions']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = PassiveAggressiveRegressor()

model.fit(xtrain, ytrain)

y_pred = model.predict(xtest)

mse = mean_squared_error(ytest, y_pred)
print("Mean Squared Error (MSE):", mse)

mae = mean_absolute_error(ytest, y_pred)
print("Mean Absolute Error (MAE):", mae)

r2 = r2_score(ytest, y_pred)
print("R-squared (R2):", r2)

new_features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
prediction = model.predict(new_features)
print("Predicted Impressions:", prediction)