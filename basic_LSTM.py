# %%
import yfinance as yf
import pandas as pd
import numpy as np

# %%

ticker = "MSFT"

df = pd.DataFrame(yf.download(tickers=ticker, period="2y")["Close"])
df = df.rename(columns={"Close": "Target"})

# %%
import matplotlib.pyplot as plt

plt.plot(df)



# %%
def df_to_windowed_df(dataframe, n):
    for i in range(1, n + 1):
        dataframe["Target" + str(-i)] = dataframe["Target"].shift(i)

    dataframe.dropna(axis=0, inplace=True)

    return dataframe


windowed_df = df_to_windowed_df(df, 2)


# %%
def windowed_df_to_date_X_y(windowed_df):
    values = windowed_df.to_numpy()
    dates = windowed_df.index.to_numpy()
    matrix = values[:, 1:]
    X = matrix.reshape((len(dates), matrix.shape[1], 1))

    y = values[:, 0]

    return dates, X.astype(np.float32), y.astype(np.float32)


dates, X, Y = windowed_df_to_date_X_y(windowed_df)
# %%
q_80 = int(len(dates) * 0.8)
q_90 = int(len(dates) * 0.9)
# %%
dates_train, X_train, y_train = dates[:q_80], X[:q_80], Y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], Y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], Y[q_90:]


plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)

plt.legend(["Train", "Validation", "Test"])
# %%
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras import layers

model = Sequential(
    [
        layers.Input((X_train.shape[1], 1)),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ]
)

model.compile(
    loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mean_absolute_error"]
)

# %%
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
# %%
train_predictions = model.predict(X_train).flatten()
val_predictions = model.predict(X_val).flatten()
test_predictions = model.predict(X_test).flatten()
plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.plot(dates_val, val_predictions)
plt.plot(dates_val, y_val)
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(
    [
        "Training Predictions",
        "Training Observations",
        "Validation Predictions",
        "Validation Observations",
        "Testing Predictions",
        "Testing Observations",
    ]
)


"""Too much data, not good at extrapolating if the data is out of range,
   train on data within the range"""
# %%
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(
    [
        "Testing Predictions",
        "Testing Observations",
    ]
)

# %%
