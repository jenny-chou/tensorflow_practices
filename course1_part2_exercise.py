import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=[1])])# Your Code Here#
model.compile(optimizer='sgd', loss='mse')# Your Code Here#)
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0]).astype('float')# Your Code Here#
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0]).astype('float')# Your Code Here#
model.fit(xs, ys, epochs=500)# Your Code here#)
print(model.predict([7.0]))