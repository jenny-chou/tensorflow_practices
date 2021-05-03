import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def trend(time, slope):
    return time * slope


def seasonality(time, period, amplitude=10):
    season = (time % period) / period
    season_pattern = np.where(season<0.4, np.cos(season*2*np.pi), 1/np.exp(season*3))
    return season_pattern * amplitude


def noise(time, noise_level=1):
    return np.random.randn(len(time)) * noise_level


def window_dataset(series, window_size, batch_size, shuffle_buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(lambda row: (row[:-1], row[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def model_forecast(model, series, window_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(32).prefetch(1)
    return model.predict(dataset)


def model1():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda row: tf.expand_dims(row, axis=-1), input_shape=[None]),
        tf.keras.layers.Conv1D(64, 3, padding='causal', activation='relu'),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda row: row * 200)
    ])
    return model


def model2():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda row: tf.expand_dims(row, axis=-1), input_shape=[None]),
        tf.keras.layers.Conv1D(128, 3, padding='causal', activation='relu'),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda row: row * 200)
    ])
    return model



time = np.arange(4*365 + 1, dtype='float32')
series = 10 + trend(time, 0.4) + seasonality(time, 365, 20) + noise(time, 2)
split_time = 1000
x_train, time_train = np.array(series[:split_time]), np.array(time[:split_time])
x_valid, time_valid = np.array(series[split_time:]), np.array(time[split_time:])
# plt.figure(figsize=(10,6))
# plt.plot(time_train, x_train,  'r', label="Training set")
# plt.plot(time_valid, x_valid, 'b', label="Testing set")
# plt.legend()

window_size = 20
batch_size = 32
buffer_size = 1000
train_set = window_dataset(x_train, window_size, batch_size, buffer_size)

model = model2()
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-7*10**(epoch/20))
#  for model1:
# model.compile(optimizer=tf.keras.optimizers.SGD(lr=7e-7, momentum=0.9, decay=1e-6), loss=tf.keras.losses.Huber())
model.compile(optimizer=tf.keras.optimizers.SGD(lr=7e-7, momentum=0.9, decay=1e-6), loss=tf.keras.losses.Huber())
model.summary()
epoch = 300
# history = model.fit(train_set, epochs=epoch, callbacks=[lr_scheduler])
history = model.fit(train_set, epochs=epoch)

# plt.figure(figsize=(10,6))
# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 100])
# plt.xlabel("learning rate")
# plt.ylabel("loss")

predict = model_forecast(model, series[split_time-window_size:-1], window_size)
predict = np.array(predict)[:,0]
print(tf.metrics.mean_squared_error(x_valid, predict).numpy())
# model1: 12216.213
# model2:
plt.figure(figsize=(10,6))
plt.plot(time_valid, x_valid, 'r', label="x_valid")
plt.plot(time_valid, predict, 'b', label="Predict")
plt.legend()

print(history.history.keys())
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'])
# plt.plot()
# plt.legend()

plt.show()