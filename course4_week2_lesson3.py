import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def plot_series(time, series, title=""):
    plt.figure(figsize=(10, 6))
    plt.plot(time, series)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.grid(True)
    plt.title(title)


def trend(time, slope=0):
    return slope * time


def seasonality(time, period, amplitude=1):
    season_time = (time % period) / period  # divided by period to have range in 0~1
    season_pattern = np.where(season_time < 0.4, np.cos(season_time*2*np.pi), 1/np.exp(3*season_time))
    return amplitude * season_pattern


def noise(time, noise_level=1):
    return np.random.randn(len(time)) * noise_level


def windowed_dataset(series, window_size, batch_size, shuffle_buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


time = np.arange(4*365+1, dtype='float32')
series = 10 + trend(time, 0.01) + seasonality(time, 365, 40) + noise(time, 2)
print(np.array(series).shape)  # (1461,)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
print(len(x_train), len(x_valid))  # 1000 461

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

X, Y = [], []
for x, y in dataset:
    X.append(x)
    Y.append(y)
print(np.array(X[-1]).shape, np.array(Y[-1]).shape)  # (20, 20) (20,)
print(np.array(X[:-1]).shape, np.array(Y[:-1]).shape)  # (30, 32, 20) (30, 32)

model = keras.models.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=[window_size]),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1)
])
model.compile(optimizer=tf.optimizers.SGD(lr=1e-6, momentum=0.9), loss='mse')
model.fit(dataset, epochs=100)

forecast = []
for time in range(len(time_valid)):
    predict = model.predict(series[split_time-window_size+time:split_time+time][None])
    forecast.append(predict)
result = np.array(forecast)[:,0,0]

print(keras.metrics.mean_squared_error(x_valid, result).numpy())  # 20.698925

plt.figure(figsize=(10,6))
plt.plot(time_valid, x_valid)
plt.plot(time_valid, result)
plt.show()