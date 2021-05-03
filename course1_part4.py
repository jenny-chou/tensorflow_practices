import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow import keras


np.set_printoptions(precision=3)


mnist = keras.datasets.fashion_mnist
(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()
"""
np.set_printoptions(linewidth=200)
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])
"""

# try training without normalizing data
model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, shuffle=False)
model.evaluate(testing_images, testing_labels)
predict_prob = model.predict(testing_images)
print(predict_prob[0], testing_labels[0])
"""
Epoch 5/5
1875/1875 [==============================] - 8s 4ms/step - loss: 0.4864 - accuracy: 0.8372
313/313 [==============================] - 1s 2ms/step - loss: 0.5341 - accuracy: 0.8206
[8.9011737e-19 1.3133497e-14 6.4475554e-32 2.5922574e-19 6.0424773e-30
 6.8629321e-05 2.2465973e-27 6.2285747e-02 6.0840784e-18 9.3764561e-01] 9
"""


# normalize data
training_images = np.array(training_images / 255).astype(float)
testing_images = np.array(testing_images / 255).astype(float)

# train with normalized data with vanilla perceptron model
model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, shuffle=False)
model.evaluate(testing_images, testing_labels)
predict_prob = model.predict(testing_images)
print(predict_prob[0], testing_labels[0])
"""
Epoch 5/5
1875/1875 [==============================] - 2s 887us/step - loss: 0.2924 - accuracy: 0.8923
313/313 [==============================] - 0s 635us/step - loss: 0.3686 - accuracy: 0.8688
[1.9301760e-05 2.9485946e-07 8.8695806e-06 4.3017664e-07 9.0564890e-06
 1.2784616e-02 8.3296263e-06 2.3189171e-01 5.5867300e-04 7.5471872e-01] 9
"""


# train with normalized data with deeper model
model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, shuffle=False)
model.evaluate(testing_images, testing_labels)
predict_prob = model.predict(testing_images)
print(predict_prob[0], testing_labels[0])
"""
Epoch 5/5
1875/1875 [==============================] - 8s 4ms/step - loss: 0.2712 - accuracy: 0.8999
313/313 [==============================] - 1s 2ms/step - loss: 0.3569 - accuracy: 0.8758
[1.1728500e-05 1.3246369e-06 1.4492829e-06 8.9289176e-07 5.2879313e-06
 8.0818031e-03 9.2979817e-06 4.8468564e-02 1.5774101e-05 9.4340390e-01] 9
"""


# stop training after the epoch which reached certain accuracy or loss
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # if(logs.get('loss') < 0.4):
        if logs.get('accuracy') > 0.6:
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True

model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5, shuffle=False, callbacks=[myCallback()])
model.evaluate(testing_images, testing_labels)
predict_prob = model.predict(testing_images)
print(predict_prob[0], testing_labels[0])
"""
Epoch 1/5
1869/1875 [============================>.] - ETA: 0s - loss: 0.4677 - accuracy: 0.8315
Reached 60% accuracy so cancelling training!
1875/1875 [==============================] - 8s 4ms/step - loss: 0.4673 - accuracy: 0.8316
313/313 [==============================] - 1s 2ms/step - loss: 0.4172 - accuracy: 0.8461
[2.076e-05 2.087e-06 1.190e-05 5.014e-06 9.295e-06 5.623e-02 1.190e-05
 2.344e-01 5.137e-04 7.088e-01] 9
"""