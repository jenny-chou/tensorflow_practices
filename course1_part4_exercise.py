# YOUR CODE SHOULD START HERE
import numpy as np
# YOUR CODE SHOULD END HERE
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# YOUR CODE SHOULD START HERE
print(x_train.shape)
x_train = np.array(tf.expand_dims(x_train, axis=-1)).astype('float')/255
x_test = np.array(tf.expand_dims(x_test, axis=-1)).astype('float')/255
y_train = np.array(y_train)
y_test = np.array(y_test)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

# YOUR CODE SHOULD END HERE
model = tf.keras.models.Sequential([
    # YOUR CODE SHOULD START HERE
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=x_train[0].shape),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
    # YOUR CODE SHOULD END HERE
])
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# YOUR CODE SHOULD START HERE
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[myCallback()])
# YOUR CODE SHOULD END HERE
"""
Epoch 1/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.1360 - accuracy: 0.9589 - val_loss: 0.0649 - val_accuracy: 0.9794
Epoch 2/10
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0460 - accuracy: 0.9857 - val_loss: 0.0574 - val_accuracy: 0.9829
Epoch 3/10
1874/1875 [============================>.] - ETA: 0s - loss: 0.0266 - accuracy: 0.9915
Reached 99% accuracy so cancelling training!
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0266 - accuracy: 0.9915 - val_loss: 0.0377 - val_accuracy: 0.9865
"""