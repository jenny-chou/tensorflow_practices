import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def window_dataset(series, window_size, batch_size, buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(buffer_size)
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
        tf.keras.layers.Conv1D(32, 5, padding='causal', activation='relu'),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda row: row * 400)
    ])
    return model


def model2():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda row: tf.expand_dims(row, axis=-1), input_shape=[None]),
        tf.keras.layers.Conv1D(120, 5, padding='causal', activation='relu'),
        tf.keras.layers.LSTM(120, return_sequences=True),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda row: row * 400)
    ])
    return model


sunspot, time = [], []
with open("Sunspots.csv", 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)
    for row in reader:
        sunspot.append(float(row[2]))
        time.append(float(row[0]))
sunspot = np.array(sunspot)
time = np.array(time)
# plt.figure(figsize=(10,6))
# plt.plot(time, sunspot)

split_time = 2000
x_train, time_train = sunspot[:split_time], time[:split_time]
x_valid, time_valid = sunspot[split_time:], time[split_time:]
window_size = 62
batch_size = 16
buffer_size = 1000
train_set = window_dataset(x_train, window_size, batch_size, buffer_size)

model = model2()
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8*10**(epoch/20))
model.compile(optimizer=tf.optimizers.SGD(lr=1e-5, momentum=0.9), loss=tf.losses.Huber(), metrics=['mae'])
epoch = 300
# history = model.fit(train_set, epochs=epoch, callbacks=[lr_scheduler])
history = model.fit(train_set, epochs=epoch)


# plt.figure(figsize=(10,6))
# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 5000])
# plt.xlabel("learning rate")
# plt.ylabel("loss")

predict = model_forecast(model, sunspot[split_time-window_size:-1], window_size)
predict = np.array(predict)[:,-1,0]
print(tf.metrics.mean_absolute_error(x_valid, predict).numpy())
# model1: 12216.213
# model2:
plt.figure(figsize=(10,6))
plt.plot(time_valid, x_valid, 'r', label="x_valid")
plt.plot(time_valid, predict, 'b', label="Predict")
plt.legend()

print(history.history.keys())
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'])

plt.show()
