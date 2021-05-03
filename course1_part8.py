import tensorflow as tf
import numpy as np
import os
import zipfile
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

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
    "horse-or-human",
    target_size=(300,300),
    batch_size=128,
    class_mode='binary'
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
    "validation-horse-or-human",
    target_size=(300,300),
    batch_size=32,
    class_mode='binary'
)

model = keras.models.Sequential([
    keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, input_shape=(300, 300, 3)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.optimizers.RMSprop(lr=0.001), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=8, epochs=5, validation_data=(test_generator), validation_steps=8)
"""
Epoch 1/5
8/8 [==============================] - 37s 5s/step - loss: 6.6544 - accuracy: 0.5244 - val_loss: 0.6103 - val_accuracy: 0.5703
Epoch 2/5
8/8 [==============================] - 31s 4s/step - loss: 0.7741 - accuracy: 0.7686 - val_loss: 0.6495 - val_accuracy: 0.6641
Epoch 3/5
8/8 [==============================] - 31s 4s/step - loss: 0.1543 - accuracy: 0.9633 - val_loss: 1.8845 - val_accuracy: 0.6602
Epoch 4/5
8/8 [==============================] - 33s 4s/step - loss: 0.2017 - accuracy: 0.9229 - val_loss: 0.8070 - val_accuracy: 0.8281
Epoch 5/5
8/8 [==============================] - 30s 4s/step - loss: 0.0431 - accuracy: 0.9878 - val_loss: 1.5335 - val_accuracy: 0.7734
"""