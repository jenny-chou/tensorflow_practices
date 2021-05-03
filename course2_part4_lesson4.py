import tensorflow as tf
import numpy as np
import os
import zipfile
import random
from tensorflow import keras

"""
zip_file_path = "horse-or-human.zip"
zip_ref = zipfile.ZipFile(zip_file_path, 'r')
zip_ref.extractall("horse-or-human/")
zip_ref.close()

zip_file_path = "validation-horse-or-human.zip"
zip_ref = zipfile.ZipFile(zip_file_path, 'r')
zip_ref.extractall("validation-horse-or-human/")
zip_ref.close()
"""

train_horse_dir = os.path.join("horse-or-human", "horses")
train_human_dir = os.path.join("horse-or-human", "humans")
test_horse_dir = os.path.join("validation-horse-or-human", "horses")
test_human_dir = os.path.join("validation-horse-or-human", "humans")

print("total train horses and humans:", len(os.listdir(train_horse_dir)), len(os.listdir(train_human_dir)))
print("total test horses and humans:", len(os.listdir(test_horse_dir)), len(os.listdir(test_human_dir)))

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    "horse-or-human",
    target_size=(150,150),
    batch_size=128,
    class_mode='binary'
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
    "validation-horse-or-human",
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

model = keras.models.Sequential([
    keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, input_shape=(150, 150, 3)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.optimizers.RMSprop(lr=0.001), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
epochs = 100
history = model.fit(train_generator, steps_per_epoch=8, epochs=epochs, validation_data=(test_generator), validation_steps=8)
