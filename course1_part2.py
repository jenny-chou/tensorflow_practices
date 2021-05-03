import tensorflow as tf
import numpy as np
from tensorflow import keras

x = np.array([-1, 0, 1, 2, 3, 4]).astype(float)
y = np.array([-3, -1, 1, 3, 5, 7]).astype(float)

model = keras.models.Sequential([keras.layers.Dense(1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=500)
print(model.predict([10.0, 11.0]))
