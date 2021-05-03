import tensorflow as tf
import os
import zipfile


DESIRED_ACCURACY = 0.999


zip_ref = zipfile.ZipFile("happy-or-sad.zip", 'r')
zip_ref.extractall("happy-or-sad")
zip_ref.close()

class myCallback(tf.keras.callbacks.Callback):# Your Code):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>DESIRED_ACCURACY):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True
  # Your Code

callbacks = myCallback()

# This Code Block should Define and Compile the Model
model = tf.keras.models.Sequential([
# Your Code Here
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer='adam', loss=tf.losses.binary_crossentropy, metrics=['accuracy'])  # Your Code Here #)

# This code block should create an instance of an ImageDataGenerator called train_datagen
# And a train_generator by calling train_datagen.flow_from_directory

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)# Your Code Here

train_generator = train_datagen.flow_from_directory(
        # Your Code Here)
    "happy-or-sad/",
    target_size=(150,150),
    batch_size=10,
    class_mode='binary'
)

# Expected output: 'Found 80 images belonging to 2 classes'

# This code block should call model.fit and train for
# a number of epochs.
history = model.fit(train_generator, steps_per_epoch=8, epochs=10, callbacks=[myCallback()])
# Your Code Here)

# Expected output: "Reached 99.9% accuracy so cancelling training!""
"""
Epoch 6/10
8/8 [==============================] - 0s 13ms/step - loss: 0.0465 - accuracy: 0.9875
Epoch 7/10
5/8 [=================>............] - ETA: 0s - loss: 0.0298 - accuracy: 1.0000
Reached 99.9% accuracy so cancelling training!
8/8 [==============================] - 0s 13ms/step - loss: 0.0225 - accuracy: 1.0000
"""