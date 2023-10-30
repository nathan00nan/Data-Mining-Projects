import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as pyo
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

data = pd.read_csv(r"C:\Users\pumpk\Desktop\Nathan's File\DOCUMENTS\Hong Kong\PolyU\2023 FALL\LGT5083\Individual Assignment\Data.csv")

data["Date"] = pd.to_datetime(data["date"], format='%Y-%m-%d')

# print(data.head())
# print(data.describe())
# print(data.info())

data = data.rename(columns={"Date": "ds", "Spend Amount": "y"})

model = Prophet(
    holidays=pd.DataFrame({
        'holiday': 'custom_holiday',
        'ds': pd.to_datetime(['2023-01-01', '2023-12-25']),
        'lower_window': 0,
        'upper_window': 1,
    }),
    seasonality_prior_scale=0.1,
    holidays_prior_scale=10.0,
    changepoint_prior_scale=0.05,
)
model.add_seasonality(name='weekly', period=7, fourier_order=3)

model.fit(data)

forecast_periods = 365
forecasts = model.make_future_dataframe(periods=forecast_periods)

predictions = model.predict(forecasts)

actual_values = data['y'].values
predicted_values = predictions['yhat'][:-forecast_periods].values

mae = mean_absolute_error(actual_values, predicted_values)
mse = mean_squared_error(actual_values, predicted_values)
rmse = np.sqrt(mse)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(actual_values, predicted_values)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

fig = plot_plotly(model, predictions)
pyo.iplot(fig)