import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


time = np.arange(10 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 40
slope = 0.005
noise_level = 3

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=51)
plot_series(time, series)

split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x*100)
])
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics= ["mae"])
history = model.fit(dataset, epochs=10, callbacks=[lr_schedule])
"""
Epoch 1/10
94/94 [==============================] - 1s 9ms/step - loss: 8.5331 - mae: 9.0204
Epoch 2/10
94/94 [==============================] - 1s 8ms/step - loss: 8.0548 - mae: 8.5392
Epoch 3/10
94/94 [==============================] - 1s 8ms/step - loss: 7.7249 - mae: 8.2080
Epoch 4/10
94/94 [==============================] - 1s 8ms/step - loss: 7.5026 - mae: 7.9862
Epoch 5/10
94/94 [==============================] - 1s 8ms/step - loss: 7.3488 - mae: 7.8287
Epoch 6/10
94/94 [==============================] - 1s 8ms/step - loss: 7.2340 - mae: 7.7112
Epoch 7/10
94/94 [==============================] - 1s 8ms/step - loss: 7.1368 - mae: 7.6127
Epoch 8/10
94/94 [==============================] - 1s 8ms/step - loss: 7.0504 - mae: 7.5257
Epoch 9/10
94/94 [==============================] - 1s 14ms/step - loss: 6.9634 - mae: 7.4375
Epoch 10/10
94/94 [==============================] - 1s 8ms/step - loss: 6.8760 - mae: 7.3503
"""

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])
plt.xlabel("learning rate")
plt.ylabel("loss")
# FROM THIS PICK A LEARNING RATE

tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x*100)
])
model.compile(loss='mse', optimizer=tf.optimizers.SGD(lr=1e-8, momentum=0.9), metrics=['mae'])
history = model.fit(dataset, epochs=10)
# FIND A MODEL AND A LR THAT TRAINS TO AN MAE < 3
"""
Epoch 1/10
94/94 [==============================] - 3s 30ms/step - loss: 662.9711 - mae: 17.9182
Epoch 2/10
94/94 [==============================] - 3s 30ms/step - loss: 125.0181 - mae: 8.1419
Epoch 3/10
94/94 [==============================] - 3s 29ms/step - loss: 78.4763 - mae: 6.2838
Epoch 4/10
94/94 [==============================] - 3s 30ms/step - loss: 63.2349 - mae: 5.4968
Epoch 5/10
94/94 [==============================] - 3s 30ms/step - loss: 55.7216 - mae: 5.1223
Epoch 6/10
94/94 [==============================] - 3s 29ms/step - loss: 51.9345 - mae: 4.9167
Epoch 7/10
94/94 [==============================] - 3s 30ms/step - loss: 49.6900 - mae: 4.7802
Epoch 8/10
94/94 [==============================] - 3s 30ms/step - loss: 47.4935 - mae: 4.6337
Epoch 9/10
94/94 [==============================] - 3s 30ms/step - loss: 46.0241 - mae: 4.5796
Epoch 10/10
94/94 [==============================] - 3s 30ms/step - loss: 44.9066 - mae: 4.5029
"""

forecast, result = [], []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())  # 4.625748
# YOUR RESULT HERE SHOULD BE LESS THAN 4

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
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

plt.show()
