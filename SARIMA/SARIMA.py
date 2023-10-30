import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv(r"C:\Users\pumpk\Desktop\Nathan's File\DOCUMENTS\My Notes\Projects\SARIMA.csv")

data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")

p, d, q = 5, 1, 2

P, D, Q, S = 1, 1, 1, 12

model = sm.tsa.statespace.SARIMAX(data['Views'], order=(p, d, q), seasonal_order=(P, D, Q, S))
model = model.fit()

predictions = model.get_prediction()

predicted_values = predictions.predicted_mean

actual_values = data['Views']

mae = mean_absolute_error(actual_values, predicted_values)

rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

r2 = r2_score(actual_values, predicted_values)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R^2): {r2}")

future_predictions = model.predict(len(data), len(data)+50)
print(future_predictions)

data["Views"].plot(legend=True, label="Training Data", figsize=(15, 10))
future_predictions.plot(legend=True, label="Predictions")