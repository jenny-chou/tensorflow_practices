# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 09:07:51 2020

@author: jenny
"""

import os
import zipfile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# # extract zip file
# # # dataset found here: http://www.laurencemoroney.com/rock-paper-scissors-dataset/
# zip_file = "rps.zip"
# zip_ref = zipfile.ZipFile(zip_file, 'r')
# zip_ref.extractall()
# zip_ref.close()

# zip_file = "rps-test-set.zip"
# zip_ref = zipfile.ZipFile(zip_file, 'r')
# zip_ref.extractall()
# zip_ref.close()

# use CNN with rock/paper/sissor dataset
train_dir = "rps/"

# ImageDataGenerator is a image generator that generates new augmented images from original
# images at RUNTIME.
# First, create an image generator object and specify how to generate image:
# Ex: rotate, whitening, shear, shift, flip, rescale, ...
train_datagen = ImageDataGenerator(
    rescale=1./255,
    featurewise_center=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

# Then, load the original image from directory and subdirectories:
# SHOULD point to directory that has subdirectories that has all the labelled data.
# Name of subdirectory should be the label name so that ImageDataGenerator can automatically
# generate image+label dataset:
# EX:
# DIR---TRAIN----Label1----image1.jpg, image2.jpg, ...
#  |        |----Label2----ImgA.jpg, ImgB.jpg, ...
#  |----TEST-----Label1----red.jpg, yel.jpg, ...
#           |----Label2----A.jpg, B.jpg, ...
#
# flow_from_directory(DIR, ...) to load image from OS
# flow(X, Y, ...) to load image in IDE
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Points to the dir that has the labelled subdirectories
    target_size=(150,150),  # Original image size is (300, 300), resized to (150, 150)
                            # at runtime when loaded and not affecting the original image
    class_mode='categorical'   # others: "binary"
#     batch_size=128   # images are loaded in batches during training and testing,
#                      # which is more efficient than loading image one by one
)

test_dir = "rps-test-set/"
# Test data just need to normalized/rescale to fit with model
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    class_mode='categorical'
)

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.01),
              metrics=['accuracy'])
model.summary()
"""
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 148, 148, 64)      1792      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 74, 74, 64)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 72, 72, 64)        36928     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 34, 34, 128)       73856     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 15, 15, 128)       147584    
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6272)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 6272)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               3211776   
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 1539      
=================================================================
Total params: 3,473,475
Trainable params: 3,473,475
Non-trainable params: 0
"""
history = model.fit(train_generator, epochs=15, validation_data=test_generator)
"""
Epoch 15/15
79/79 [==============================] - 38s 481ms/step 
- loss: 0.0701 - accuracy: 0.9786 - val_loss: 0.0413 - val_accuracy: 0.9812
"""
"""
# Alternative training method:
# ImageDataGenerator loads images in batches of size=batch_size.
# steps_per_epoch in model.fit() is the number of batches to yield from generator
# before declaring one epoch finished.
# --> steps_per_epoch = ceil( len(training_set)/batch_size )
#
history = model.fit(train_generator, steps_per_epoch=8, epochs=15,
                    validation_data=test_generator, validation_steps=8)

# Here's a more "manual" example without using iterator
for e in range(epochs):
    batches = 0
    for x_batch, y_batch in train_datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
"""

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()