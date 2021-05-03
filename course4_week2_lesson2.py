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
print(type(dataset))  # <class 'tensorflow.python.data.ops.dataset_ops.PrefetchDataset'>
print(dataset)  # <PrefetchDataset shapes: ((None, None), (None,)), types: (tf.float64, tf.float64)>
# Tensor's element spec:
#  (TensorSpec(shape=(None, None), dtype=tf.float64, name=None),
#   TensorSpec(shape=(None,),      dtype=tf.float64, name=None))

X, Y = [], []
for x, y in dataset:
    X.append(x)
    Y.append(y)
print(np.array(X).shape, np.array(Y).shape)  # (31,) (31,)
print(np.array(X[:-1]).shape, np.array(Y[:-1]).shape)  # (30, 32, 20) (30, 32)

print(X[-1], Y[-1])
# X[0] = tf.Tensor( ... , shape=(32, 20), dtype=float64)
# Y[0] = tf.Tensor( ... , shape=(32,), dtype=float64)
# X[-1] = tf.Tensor( ... , shape=(20, 20), dtype=float64)
# Y[-1] = tf.Tensor( ... , shape=(20,), dtype=float64)

layer = keras.layers.Dense(1, input_shape=[window_size])
model = keras.models.Sequential([layer])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9), loss='mse')
model.fit(dataset, epochs=100)  # xs=dataset's x, ys=dataset's y, batch=batch_size
print("layer weights {}".format(layer.get_weights()))
"""
Epoch 95/100
31/31 [==============================] - 0s 1ms/step - loss: 20.4584
Epoch 96/100
31/31 [==============================] - 0s 1ms/step - loss: 20.3077
Epoch 97/100
31/31 [==============================] - 0s 1ms/step - loss: 20.2408
Epoch 98/100
31/31 [==============================] - 0s 1ms/step - loss: 20.1583
Epoch 99/100
31/31 [==============================] - 0s 2ms/step - loss: 20.1407
Epoch 100/100
31/31 [==============================] - 0s 2ms/step - loss: 20.0351

# layer's 20 weights and 1 bias
layer weights [array([[-0.07595737],
       [ 0.15894356],
       [ 0.03936027],
       [-0.28886005],
       [ 0.20654945],
       [ 0.12322352],
       [-0.28654054],
       [ 0.13690533],
       [-0.0568324 ],
       [-0.06887526],
       [ 0.10098746],
       [ 0.10769469],
       [-0.05046599],
       [-0.09387366],
       [-0.13369052],
       [ 0.06482475],
       [ 0.15807343],
       [ 0.31494984],
       [-0.016007  ],
       [ 0.6533984 ]], dtype=float32), array([0.01672014], dtype=float32)]
"""

forecast = []
for time in range(len(series)-window_size):
    predict = model.predict(series[time:time+window_size][np.newaxis])  # needs to have same format as dataset
    # np.newaxis can be used in all slicing operations to create  an axis of length 1. Alias of None:
    # predict = model.predict(series[time:time+window_size][None]) # yield same result as np.newaxis
    # print(np.array(predict).shape)  # (1, 1)
    forecast.append(predict)
forecast = forecast[split_time-window_size:]  # forecast = [ [[111]], [[222]], [[333]], [[444]], ... ]
result = np.array(forecast)[:,0,0]
print(type(forecast))  # <class 'list'>
print(np.array(forecast).shape)  # (461, 1, 1)
print(result.shape)  # (461,)

plt.figure()
plt.plot(time_valid, x_valid)
plt.plot(time_valid, result)
plt.show()

print(keras.metrics.mean_absolute_error(x_valid, result).numpy())  # 2.4023886
