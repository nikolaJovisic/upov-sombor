import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape

from dataset import OUTPUT_COLUMNS, OUTPUT_COLUMN, load, preprocess

data = load()
train_data, validation_data = preprocess(data, use_seq=False)

# train_size = int(0.8 * len(data))
#
# train_data = data[:train_size]
# validation_data = data[train_size:]

X_train = train_data.drop(columns=OUTPUT_COLUMNS)
y_train = train_data[OUTPUT_COLUMN]
X_val = validation_data.drop(columns=OUTPUT_COLUMNS)
y_val = validation_data[OUTPUT_COLUMN]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

r2 = r2_score(y_val, y_pred)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))


print("R-squared (R2):", r2)
print("Root Mean Squared Error (RMSE):", rmse)
print('mape: ', mape(y_val, y_pred))
# for i in zip(y_pred, y_val):
#     print(i)

plt.scatter(y_pred, y_val)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.show()
