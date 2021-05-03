import tensorflow as tf
import numpy as np
from tensorflow import keras

(training_images, training_labels), (testing_images, testing_labels) = keras.datasets.fashion_mnist.load_data()
training_images = np.expand_dims(training_images, axis=3)
training_images = training_images/255
testing_images = np.expand_dims(testing_images, axis=3)
testing_images = testing_images/255

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(training_images[0].shape)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, batch_size=10, epochs=5)
model.evaluate(testing_images, testing_labels)
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 32)        9248      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 32)          0         
_________________________________________________________________
flatten (Flatten)            (None, 800)               0         
_________________________________________________________________
dense (Dense)                (None, 512)               410112    
_________________________________________________________________
dense_1 (Dense)              (None, 10)                5130      
=================================================================
Total params: 424,810
Trainable params: 424,810
Non-trainable params: 0
_________________________________________________________________
Epoch 5/5
6000/6000 [==============================] - 35s 6ms/step - loss: 0.1763 - accuracy: 0.9335
313/313 [==============================] - 1s 3ms/step - loss: 0.2843 - accuracy: 0.9053
"""


model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(training_images[0].shape)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(training_images, training_labels, batch_size=10, epochs=5)
model.evaluate(testing_images, testing_labels)
"""
Epoch 5/5
6000/6000 [==============================] - 87s 15ms/step - loss: 0.1151 - accuracy: 0.9574
313/313 [==============================] - 1s 5ms/step - loss: 0.3312 - accuracy: 0.9089
"""

model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(training_images[0].shape)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(training_images, training_labels, batch_size=10, epochs=5)
model.evaluate(testing_images, testing_labels)
"""
Epoch 5/5
6000/6000 [==============================] - 168s 28ms/step - loss: 0.1044 - accuracy: 0.9617
313/313 [==============================] - 2s 7ms/step - loss: 0.3285 - accuracy: 0.9115
"""

model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(training_images[0].shape)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(training_images, training_labels, batch_size=10, epochs=5)
model.evaluate(testing_images, testing_labels)
"""
Epoch 5/5
6000/6000 [==============================] - 68s 11ms/step - loss: 0.1667 - accuracy: 0.9368
313/313 [==============================] - 2s 6ms/step - loss: 0.3135 - accuracy: 0.8951
"""

model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(training_images[0].shape)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(training_images, training_labels, batch_size=10, epochs=5)
model.evaluate(testing_images, testing_labels)
"""
Epoch 5/5
6000/6000 [==============================] - 48s 8ms/step - loss: 0.2627 - accuracy: 0.9029
313/313 [==============================] - 2s 6ms/step - loss: 0.3288 - accuracy: 0.8827
"""