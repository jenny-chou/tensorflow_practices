import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def window_series(series, window_size, batch_size, buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(lambda row: (row[:-1], row[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def model1():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda row: tf.expand_dims(row, axis=-1), input_shape=[None]),
        tf.keras.layers.Conv1D(120, 5, padding='causal', activation='relu'),
        tf.keras.layers.LSTM(120, return_sequences=True),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda row: row * 400)
    ])
    return model


def model_forecast(model, series, window_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(32).prefetch(1)
    return model.predict(dataset)


time, temp = [], []
with open("daily-min-temp.csv", 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader)
    time_count = 1
    for row in reader:
        time.append(time_count)
        time_count += 1
        temp.append(float(row[1]))
time, temp = np.array(time), np.array(temp)

split_size = 2500
x_train, time_train = np.array(temp[:split_size]), np.array(time[:split_size])
x_valid, time_valid = np.array(temp[split_size:]), np.array(time[split_size:])
window_size = 30
batch_size = 32
buffer_size = 1000
train_set = window_series(x_train, window_size, batch_size, buffer_size)

model = model1()
model.summary()
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))
model.compile(optimizer=tf.optimizers.RMSprop(lr=1e-6, momentum=0.9), loss=tf.losses.Huber(), metrics=['mae'])
epochs = 100
history = model.fit(train_set, epochs=epochs)  #, callbacks=[lr_scheduler])
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda (Lambda)              (None, None, 1)           0         
_________________________________________________________________
conv1d (Conv1D)              (None, None, 120)         720       
_________________________________________________________________
lstm (LSTM)                  (None, None, 120)         115680    
_________________________________________________________________
lstm_1 (LSTM)                (None, None, 60)          43440     
_________________________________________________________________
dense (Dense)                (None, None, 40)          2440      
_________________________________________________________________
dense_1 (Dense)              (None, None, 20)          820       
_________________________________________________________________
dense_2 (Dense)              (None, None, 1)           21        
_________________________________________________________________
lambda_1 (Lambda)            (None, None, 1)           0         
=================================================================
Total params: 163,121
Trainable params: 163,121
Non-trainable params: 0
_________________________________________________________________
Epoch 98/100
78/78 [==============================] - 1s 8ms/step - loss: 1.9807 - mae: 2.4358
Epoch 99/100
78/78 [==============================] - 1s 8ms/step - loss: 1.9736 - mae: 2.4273
Epoch 100/100
78/78 [==============================] - 1s 8ms/step - loss: 1.9843 - mae: 2.4389
"""

# plt.figure(figsize=(10,6))
# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 20])
# plt.xlabel("learning rate")
# plt.ylabel("loss")

predict = model_forecast(model, temp[split_size-window_size:-1], window_size)
predict = np.array(predict)[:,-1,0]
print(tf.metrics.mean_absolute_error(x_valid, predict).numpy())  # 1.9624004
plt.figure(figsize=(10,6))
plt.plot(time_valid, x_valid, 'r', label="x_valid")
plt.plot(time_valid, predict, 'b', label="Predict")
plt.legend()

plt.show()
