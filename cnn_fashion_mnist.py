import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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
model.fit(training_images, training_labels, batch_size=10, epochs=1)
model.evaluate(testing_images, testing_labels)

fig, ax_arr = plt.subplots(3,5)
images = [0,1,2]
convolution_number = 1
model_layers_outputs = [layer.output for layer in model.layers]
model_input = model.input
activation_model = keras.models.Model(inputs=model_input, outputs=model_layers_outputs)
for layer in range(0,4):
    tmp = activation_model.predict(testing_images[images[0]].reshape(1,28,28,1))
    fig1 = activation_model.predict(testing_images[images[0]].reshape(1,28,28,1))[layer]
    ax_arr[0,layer].imshow(fig1[0,:,:,convolution_number], cmap='inferno')
    ax_arr[0,layer].grid(False)
    fig2 = activation_model.predict(testing_images[images[1]].reshape(1,28,28,1))[layer]
    ax_arr[1, layer].imshow(fig2[0, :, :, convolution_number], cmap='inferno')
    ax_arr[1, layer].grid(False)
    fig3 = activation_model.predict(testing_images[images[2]].reshape(1,28,28,1))[layer]
    ax_arr[2, layer].imshow(fig3[0, :, :, convolution_number], cmap='inferno')
    ax_arr[2, layer].grid(False)
for x in range(0,3):
    ax_arr[x, 4].imshow(testing_images[images[x]])
    ax_arr[x, 4].grid(False)
plt.show()