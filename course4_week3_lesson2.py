import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import matplotlib.image  as mpimg


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
    dataset = dataset.batch(2).prefetch(1)
    return dataset


def plot_history(history):
    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    mae = history.history['mae']
    loss = history.history['loss']

    epochs = range(len(loss))  # Get number of epochs

    # ------------------------------------------------
    # Plot MAE and Loss
    # ------------------------------------------------
    plt.figure()
    plt.plot(epochs, mae, 'r')
    plt.plot(epochs, loss, 'b')
    plt.title('MAE and Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["MAE", "Loss"])

    epochs_zoom = epochs[200:]
    mae_zoom = mae[200:]
    loss_zoom = loss[200:]

    # ------------------------------------------------
    # Plot Zoomed MAE and Loss
    # ------------------------------------------------
    plt.figure()
    plt.plot(epochs_zoom, mae_zoom, 'r')
    plt.plot(epochs_zoom, loss_zoom, 'b')
    plt.title('MAE and Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["MAE", "Loss"])


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
batch_size = 128
shuffle_buffer_size = 1000
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

X, Y = [], []
for x, y in dataset:
    X.append(x)
    Y.append(y)
print(np.array(X[-1]).shape, np.array(Y[-1]).shape)  # (2, 20) (2,)
print(np.array(X[:-1]).shape, np.array(Y[:-1]).shape)  # (489, 2, 20) (489, 2)


model = keras.models.Sequential([
    keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    keras.layers.SimpleRNN(40, return_sequences=True),
    keras.layers.SimpleRNN(40),
    keras.layers.Dense(1),
    keras.layers.Lambda(lambda x: x*100)
])
model.compile(optimizer=tf.optimizers.SGD(lr=5e-5, momentum=0.9), loss=keras.losses.Huber(), metrics=['mae'])
history = model.fit(dataset, epochs=50)

print(history.history.keys())  # dict_keys(['loss', 'mae'])
plot_history(history)

forecast = []
for time in range(len(time_valid)):
    predict = model.predict(series[split_time-window_size+time:split_time+time][None])
    forecast.append(predict)
result = np.array(forecast)[:,0,0]

print(keras.metrics.mean_squared_error(x_valid, result).numpy())  # 136.79614

plt.figure(figsize=(10,6))
plt.plot(time_valid, x_valid)
plt.plot(time_valid, result)
plt.title("RNN & without lr scheduler")


tf.keras.backend.clear_session()

model = keras.models.Sequential([
    keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[window_size]),
    keras.layers.SimpleRNN(40, return_sequences=True),
    keras.layers.SimpleRNN(40),
    keras.layers.Dense(1),
    keras.layers.Lambda(lambda x: x*100)
])
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))
model.compile(optimizer=tf.optimizers.SGD(lr=1e-6, momentum=0.9), loss=keras.losses.Huber(), metrics=['mae'])
history = model.fit(dataset, epochs=50, callbacks=[lr_scheduler])

print(history.history.keys())  # dict_keys(['loss', 'mae', 'lr'])
plot_history(history)

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])
plt.xlabel("learning rate")
plt.ylabel("loss")

forecast = []
for time in range(len(time_valid)):
    predict = model.predict(series[split_time-window_size+time:split_time+time][None])
    forecast.append(predict)
result = np.array(forecast)[:,0,0]

print(keras.metrics.mean_absolute_error(x_valid, result).numpy())  # 2.929769

plt.figure(figsize=(10,6))
plt.plot(time_valid, x_valid)
plt.plot(time_valid, result)
plt.title("RNN & with lr scheduler")
plt.show()

