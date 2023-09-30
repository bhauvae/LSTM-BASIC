# %%
import yfinance as yf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


# %%

ticker = "AAPL"

df = pd.DataFrame(yf.download(tickers=ticker, period="1y")["Close"])
df.rename(columns={"Close": "Target"}, inplace=True)


# %%
# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values.reshape(-1, 1))



# %%
def df_to_windowed_df(dataframe, n):
    for i in range(1, n + 1):
        dataframe["Target" + str(-i)] = dataframe["Target"].shift(i)

    dataframe.dropna(axis=0, inplace=True)

    return dataframe


windowed_df = df_to_windowed_df(df, 60)


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

# %%
dates_train, X_train, y_train = dates[:q_80], X[:q_80], Y[:q_80]
dates_test, X_test, y_test = dates[q_80:], X[q_80:], Y[q_80:]


# %%
def LSTM_model():
    # Initialize a sequential model
    model = Sequential()

    # Add the first LSTM layer with 50 units, input shape, and return sequences
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.2))

    # Add a second LSTM layer with 50 units and return sequences
    model.add(LSTM(units=50, return_sequences=True))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.2))
    # Add a second LSTM layer with 50 units and return sequences
    model.add(LSTM(units=50, return_sequences=True))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.2))

    # Add a third LSTM layer with 50 units
    model.add(LSTM(units=50))
    # Add dropout to prevent overfitting
    model.add(Dropout(0.2))

    # Add a dense output layer with one unit
    model.add(Dense(units=1))

    return model


model = LSTM_model()
model.summary()
model.compile(optimizer="adam", loss="mean_squared_error")

# %%

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
)

# %%
train_predictions = model.predict(X_train).flatten()
test_predictions = model.predict(X_test).flatten()
# %%
plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.plot(dates_test, test_predictions)
plt.plot(dates_test, y_test)
plt.legend(
    [
        "Training Predictions",
        "Training Observations",
        "Testing Predictions",
        "Testing Observations",
    ]
)
# %%
