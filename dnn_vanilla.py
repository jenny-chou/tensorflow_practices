import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# simple y=2x-1 model simulation with NN
x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).astype('float')
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]).astype('float')

model = Sequential([Dense(1, input_shape=(1,))])  # regression problem usually don't use activation function
model.compile(loss='mean_squared_error', optimizer='sgd')  # regression problem don't use attribute 'metrics'
model.fit(x, y, epochs=500)

print(model.predict([3.0, 4.0, 5.0, 6.0, 7.0]))

"""
# simple 2 layers perceptron model to classify 10 fashion items 
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation=tf.nn.relu),
    Dense(10, activation=tf.nn.softmax)
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
eval = model.evaluate(test_images, test_labels)
print(model.metrics_names, eval)
"""

"""
# use CNN to classify 10 types of fashion items
train_images = np.reshape(train_images, (len(train_images), 28, 28, 1))
test_images = np.reshape(test_images, (len(test_images), 28, 28, 1))
model = Sequential([
    Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(28,28,1)),
    MaxPool2D(pool_size=(2,2)),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation=tf.nn.relu),
    Dense(10, activation=tf.nn.softmax)
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
"""

# # use CNN with rock/paper/sissor dataset
# train_dir = "rps/rps/"
# train_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150,150),
#     class_mode='categorical'
# )
#
# test_dir = "rps-test-set/rps-test-set/"
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(150,150),
#     class_mode='categorical'
# )
#
# model = Sequential([
#     Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
#     MaxPool2D(pool_size=(2,2)),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPool2D(pool_size=(2,2)),
#     Conv2D(128, (3,3), activation='relu'),
#     MaxPool2D(pool_size=(2,2)),
#     Conv2D(128, (3,3), activation='relu'),
#     MaxPool2D(pool_size=(2,2)),
#     Flatten(),
#     Dropout(0.5),
#     Dense(512, activation='relu'),
#     Dense(3, activation='softmax')
# ])
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# history = model.fit(train_generator, epochs=5, validation_data=test_generator)