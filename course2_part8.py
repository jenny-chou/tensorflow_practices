import os
import zipfile
import tensorflow as tf
import numpy as np
from tensorflow import keras
#import matplotlib.pyplot as plt

zipfile_path = os.path.join("..", "pythonProject", "rps.zip")
zipfile_ref = zipfile.ZipFile(zipfile_path, 'r')
zipfile_ref.extractall(os.path.join("..", "pythonProject", "rps"))
zipfile_ref.close()

zipfile_path = os.path.join("..", "pythonProject", "rps-test-set.zip")
zipfile_ref = zipfile.ZipFile(zipfile_path, 'r')
zipfile_ref.extractall(os.path.join("..", "pythonProject", "rps"))
zipfile_ref.close()

base_dir = os.path.join("..", "pythonProject", "rps")
train_dir = os.path.join(base_dir, "rps")
train_rock_dir = os.path.join(train_dir, "rock")
train_paper_dir = os.path.join(train_dir, "paper")
train_scissors_dir = os.path.join(train_dir, "scissors")
test_dir = os.path.join(base_dir, "rps-test-set")
test_rock_dir = os.path.join(test_dir, "rock")
test_paper_dir = os.path.join(test_dir, "paper")
test_scissors_dir = os.path.join(test_dir, "scissors")

print("Total training rock, paper, scissor images:", 
      len(os.listdir(train_rock_dir)), len(os.listdir(train_paper_dir)), len(os.listdir(train_scissors_dir)))
print("Total testing rock, paper, scissor images:", 
      len(os.listdir(test_rock_dir)), len(os.listdir(test_paper_dir)), len(os.listdir(test_scissors_dir)))

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255.0,
    horizontal_flip=True,
    shear_range=0.1,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=40, class_mode='categorical')

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
test_generator = train_datagen.flow_from_directory(test_dir, target_size=(150,150), batch_size=31, class_mode='categorical')

model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(150,150,3), padding='same'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, padding='same'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, padding='same'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, padding='same'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])
model.compile(optimizer=keras.optimizers.Adam(), loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])
model.summary()
epochs = 20
history = model.fit(train_generator, steps_per_epoch=21, epochs=epochs, validation_data=test_generator, validation_steps=4)

# plt.plot(history.history['accuracy'], range(epochs), 'r', label="Training accuracy")
# plt.plot(history.history['val_accuracy'], range(epochs), 'r', label="Validation accuracy")
# plt.title("Training and validation accuracy")
# plt.legend()
# plt.figure()
#
# plt.plot(history.history['loss'], range(epochs), 'r', label="Training loss")
# plt.plot(history.history['val_loss'], range(epochs), 'r', label="Validation loss")
# plt.title("Training and validation loss")
# plt.legend()
# plt.show()