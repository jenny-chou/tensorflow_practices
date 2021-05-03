import tensorflow as tf

# YOUR CODE STARTS HERE
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.998):
      print("\nReached 99.8% accuracy so cancelling training!")
      self.model.stop_training = True
# YOUR CODE ENDS HERE

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# YOUR CODE STARTS HERE
training_images = tf.expand_dims(training_images, axis=-1)/255
test_images = tf.expand_dims(test_images, axis=-1)/255

# YOUR CODE ENDS HERE

model = tf.keras.models.Sequential([
    # YOUR CODE STARTS HERE
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    # YOUR CODE ENDS HERE
])

# YOUR CODE STARTS HERE
model.compile(optimizer="adam", loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[myCallback()])
# YOUR CODE ENDS HERE
"""
Epoch 6/10
1875/1875 [==============================] - 5s 2ms/step - loss: 0.0078 - accuracy: 0.9976 - val_loss: 0.0441 - val_accuracy: 0.9868
Epoch 7/10
1875/1875 [==============================] - 6s 3ms/step - loss: 0.0075 - accuracy: 0.9976 - val_loss: 0.0422 - val_accuracy: 0.9883
Epoch 8/10
1869/1875 [============================>.] - ETA: 0s - loss: 0.0044 - accuracy: 0.9985
Reached 99.8% accuracy so cancelling training!
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0044 - accuracy: 0.9985 - val_loss: 0.0558 - val_accuracy: 0.9863
"""