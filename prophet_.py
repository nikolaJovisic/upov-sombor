import pandas as pd
from prophet import Prophet
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

from dataset import preprocess, load, FEATURE_COLUMNS, OUTPUT_COLUMNS

df = load()
df = preprocess(df, use_lstm=False)
df = df.drop(columns=FEATURE_COLUMNS)
df.reset_index(inplace=True)
df = df.rename(columns={OUTPUT_COLUMNS[0]: "y", "Datum" : 'ds'})

# Split the data into training and validation sets
train_size = int(0.8 * len(df))
train_data = df[:train_size]
validation_data = df[train_size:]

# Create and fit the Prophet model
model = Prophet()
model.fit(train_data)

# Make predictions on the validation set
future_dates = validation_data
forecast = model.predict(future_dates)

# Extract the actual values and predicted values
y_actual = validation_data['y'].values
y_pred = forecast['yhat'].values[:len(validation_data)]

stacked = np.squeeze(np.stack((y_actual, y_pred)))

# Calculate R2 score
r2 = r2_score(y_actual, y_pred)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

# Print evaluation metrics
print('R-squared (R2):', r2)
print('Root Mean Squared Error (RMSE):', rmse)