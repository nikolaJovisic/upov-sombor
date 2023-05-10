import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

from dataset import preprocess, load, FEATURE_COLUMNS, OUTPUT_COLUMNS

df = load()
df = preprocess(df, use_lstm=False)

# Split the data into training and validation sets
train_size = int(0.8 * len(df))
train_data = df[:train_size]
validation_data = df[train_size:]

# Prepare the input features and target variable
X_train = train_data.drop(columns=OUTPUT_COLUMNS)
y_train = train_data[OUTPUT_COLUMNS[0]]
X_val = validation_data.drop(columns=OUTPUT_COLUMNS)
y_val = validation_data[OUTPUT_COLUMNS[0]]

# Create and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Calculate R2 score
r2 = r2_score(y_val, y_pred)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

# Print evaluation metrics

print('R-squared (R2):', r2)
print('Root Mean Squared Error (RMSE):', rmse)
for i in zip(y_pred, y_val):
    print(i)