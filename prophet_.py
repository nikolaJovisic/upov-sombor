import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score

from dataset import FEATURE_COLUMNS, OUTPUT_COLUMNS, load, preprocess

df = load()
df = preprocess(df, use_lstm=False)
df = df.drop(columns=FEATURE_COLUMNS)
df.reset_index(inplace=True)
df = df.rename(columns={OUTPUT_COLUMNS[0]: "y", "Datum": "ds"})

train_size = int(0.8 * len(df))
train_data = df[:train_size]
validation_data = df[train_size:]

model = Prophet()
model.fit(train_data)

future_dates = validation_data
forecast = model.predict(future_dates)

y_actual = validation_data["y"].values
y_pred = forecast["yhat"].values[: len(validation_data)]

stacked = np.squeeze(np.stack((y_actual, y_pred)))

r2 = r2_score(y_actual, y_pred)

rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

print("R-squared (R2):", r2)
print("Root Mean Squared Error (RMSE):", rmse)
