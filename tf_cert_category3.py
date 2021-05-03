# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Computer Vision with CNNs
#
# For this exercise, build and train a cats v dogs classifier
# using the Cats v Dogs dataset from TFDS.
# Be sure to use the final layer as shown
#     (Dense, 2 neurons, softmax activation)
#
# The testing infrastructure will resize all images to 224x224
# with 3 bytes of color depth. Make sure your input layer trains
# images to that specification, or the tests will fail.
#
# Make sure your output layer is exactly as specified here, or the
# tests will fail.


import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


dataset_name = 'cats_vs_dogs'
dataset, info = tfds.load(name=dataset_name, split=tfds.Split.TRAIN, with_info=True)

def preprocess(features):
    image = tf.image.resize(features['image'], [224, 224])
    image = tf.divide(image, 255)
    label = features['label']
    label = tf.cast(label, tf.float32)
    return image, label

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

def solution_model():
    train_dataset = dataset.map(preprocess).batch(32)

    pre_trained_model = tf.keras.applications.inception_v3.InceptionV3(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    for layer in pre_trained_model.layers:
        layer.trainable = False

    pre_trained_model_last_layer = pre_trained_model.get_layer("mixed7")
    pre_trained_model_last_layer_output = pre_trained_model_last_layer.output

    last_layer_outputs = tf.keras.layers.Flatten()(pre_trained_model_last_layer_output)
    last_layer_outputs = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(last_layer_outputs)
    last_layer_outputs = tf.keras.layers.Dropout(0.2)(last_layer_outputs)
    last_layer_outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(last_layer_outputs)
    model = tf.keras.models.Model(inputs=pre_trained_model.input, outputs=last_layer_outputs)
    model.compile(optimizer=tf.optimizers.RMSprop(lr=0.0001),
                  loss=tf.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    model.fit(train_dataset, steps_per_epoch=100, epochs=2, callbacks=[myCallback()])

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
