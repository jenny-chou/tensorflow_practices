import os
import zipfile
import tensorflow as tf
import numpy as np
from tensorflow import keras

zip_file_path = "cats_and_dogs_filtered.zip"
zip_ref = zipfile.ZipFile(zip_file_path, 'r')
zip_ref.extractall("cats_and_dogs")
zip_ref.close()

base_dir = "cats_and_dogs\cats_and_dogs_filtered"
train_dir = os.path.join(base_dir, "train")
train_dog_dir = os.path.join(train_dir, "dogs")
train_cat_dir = os.path.join(train_dir, "cats")
test_dir = os.path.join(base_dir, "validation")
test_dog_dir = os.path.join(test_dir, "dogs")
test_cat_dir = os.path.join(test_dir, "cats")

print("Total train dog and cat images:", len(os.listdir(train_dog_dir)), len(os.listdir(train_cat_dir)))
print("Total test dog and cat images:", len(os.listdir(test_dog_dir)), len(os.listdir(test_cat_dir)))

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
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
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu, input_shape=(150,150,3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer=tf.optimizers.RMSprop(lr=0.0001), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=100, epochs=100, validation_data=test_generator, validation_steps=50)
"""
Epoch 1/5
100/100 [==============================] - 30s 305ms/step - loss: 0.6945 - accuracy: 0.5865 - val_loss: 0.6393 - val_accuracy: 0.6190
Epoch 2/5
100/100 [==============================] - 28s 278ms/step - loss: 0.6087 - accuracy: 0.6750 - val_loss: 0.6083 - val_accuracy: 0.6600
Epoch 3/5
100/100 [==============================] - 29s 286ms/step - loss: 0.5082 - accuracy: 0.7480 - val_loss: 0.6457 - val_accuracy: 0.6550
Epoch 4/5
100/100 [==============================] - 30s 303ms/step - loss: 0.4151 - accuracy: 0.8105 - val_loss: 0.6295 - val_accuracy: 0.7040
Epoch 5/5
100/100 [==============================] - 30s 301ms/step - loss: 0.2893 - accuracy: 0.8680 - val_loss: 0.7429 - val_accuracy: 0.7010
"""