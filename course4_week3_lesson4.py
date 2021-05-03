import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import matplotlib.image  as mpimg
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


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

    plt.figure()
    plt.plot(epochs, mae, 'r')
    plt.title('MAE')
    plt.xlabel("Epochs")
    plt.ylabel("MAE")

    plt.figure()
    plt.plot(epochs, loss, 'b')
    plt.title('Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")


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
    keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(1),
    keras.layers.Lambda(lambda x: x*100)
])
model.compile(optimizer=tf.optimizers.SGD(lr=1e-5, momentum=0.9), loss='mse', metrics=['mae'])
history = model.fit(dataset, epochs=30)
"""
Epoch 20/30
490/490 [==============================] - 4s 9ms/step - loss: 122.7488 - mae: 8.0225
Epoch 21/30
490/490 [==============================] - 4s 9ms/step - loss: 102.5498 - mae: 7.5348
Epoch 22/30
490/490 [==============================] - 4s 9ms/step - loss: 170.6393 - mae: 9.7951
Epoch 23/30
490/490 [==============================] - 4s 9ms/step - loss: 135.5091 - mae: 8.5572
Epoch 24/30
490/490 [==============================] - 4s 9ms/step - loss: 68.6845 - mae: 6.0536
Epoch 25/30
490/490 [==============================] - 4s 9ms/step - loss: 83.4668 - mae: 6.6298
Epoch 26/30
490/490 [==============================] - 4s 9ms/step - loss: 55.6041 - mae: 5.5493
Epoch 27/30
490/490 [==============================] - 4s 9ms/step - loss: 51.6097 - mae: 5.1368
Epoch 28/30
490/490 [==============================] - 4s 9ms/step - loss: 95.8183 - mae: 7.1944
Epoch 29/30
490/490 [==============================] - 4s 9ms/step - loss: 62.8192 - mae: 5.6235
Epoch 30/30
490/490 [==============================] - 4s 9ms/step - loss: 73.5683 - mae: 6.4904
"""

plot_history(history)

forecast = []
for time in range(len(time_valid)):
    predict = model.predict(series[split_time-window_size+time:split_time+time][None])
    forecast.append(predict)
result = np.array(forecast)[:,0,0]
print(keras.metrics.mean_squared_error(x_valid, result).numpy())  # 95.57679

plt.figure(figsize=(10,6))
plt.plot(time_valid, x_valid)
plt.plot(time_valid, result)
plt.title("RNN & without lr scheduler")


tf.keras.backend.clear_session()

model = keras.models.Sequential([
    keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(1),
    keras.layers.Lambda(lambda x: x*100)
])
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))
model.compile(optimizer=tf.optimizers.SGD(lr=1e-6, momentum=0.9), loss=keras.losses.Huber(), metrics=['mae'])
history = model.fit(dataset, epochs=30, callbacks=[lr_scheduler])
"""
Epoch 20/30
490/490 [==============================] - 4s 9ms/step - loss: 3.0820 - mae: 3.5429
Epoch 21/30
490/490 [==============================] - 4s 9ms/step - loss: 2.9747 - mae: 3.4343
Epoch 22/30
490/490 [==============================] - 4s 9ms/step - loss: 2.8878 - mae: 3.3462
Epoch 23/30
490/490 [==============================] - 4s 9ms/step - loss: 2.7963 - mae: 3.2497
Epoch 24/30
490/490 [==============================] - 5s 9ms/step - loss: 2.7309 - mae: 3.1883
Epoch 25/30
490/490 [==============================] - 4s 9ms/step - loss: 2.6826 - mae: 3.1368
Epoch 26/30
490/490 [==============================] - 4s 9ms/step - loss: 2.6133 - mae: 3.0647
Epoch 27/30
490/490 [==============================] - 5s 9ms/step - loss: 2.5462 - mae: 3.0005
Epoch 28/30
490/490 [==============================] - 4s 9ms/step - loss: 2.4809 - mae: 2.9334
Epoch 29/30
490/490 [==============================] - 4s 9ms/step - loss: 2.4685 - mae: 2.9215
Epoch 30/30
490/490 [==============================] - 4s 9ms/step - loss: 2.4452 - mae: 2.9013
"""

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])
plt.xlabel("learning rate")
plt.ylabel("loss")
plot_history(history)

forecast = []
for time in range(len(time_valid)):
    predict = model.predict(series[split_time-window_size+time:split_time+time][None])
    forecast.append(predict)
result = np.array(forecast)[:,0,0]
print(keras.metrics.mean_absolute_error(x_valid, result).numpy())  # 3.1231322

plt.figure(figsize=(10,6))
plt.plot(time_valid, x_valid)
plt.plot(time_valid, result)
plt.title("RNN & with lr scheduler")
plt.show()

