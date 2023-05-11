import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from dataset import OUTPUT_COLUMNS, load, preprocess

df = load()
df = preprocess(df, use_seq=False)

train_size = int(0.8 * len(df))
train_data = df[:train_size]
validation_data = df[train_size:]

X_train = train_data.drop(columns=OUTPUT_COLUMNS)
y_train = train_data[OUTPUT_COLUMNS[0]]
X_val = validation_data.drop(columns=OUTPUT_COLUMNS)
y_val = validation_data[OUTPUT_COLUMNS[0]]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

r2 = r2_score(y_val, y_pred)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))


print("R-squared (R2):", r2)
print("Root Mean Squared Error (RMSE):", rmse)
for i in zip(y_pred, y_val):
    print(i)
