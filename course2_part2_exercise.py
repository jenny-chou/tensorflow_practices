# In this exercise you will train a CNN on the FULL Cats-v-dogs dataset
# This will require you doing a lot of data preprocessing because
# the dataset isn't split into training and validation for you
# This code block has all the required inputs
import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# This code block downloads the full Cats-v-Dogs dataset and stores it as
# cats-and-dogs.zip. It then unzips it to /tmp
# which will create a tmp/PetImages directory containing subdirectories
# called 'Cat' and 'Dog' (that's how the original researchers structured it)
# If the URL doesn't work,
# .   visit https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
# And right click on the 'Download Manually' link to get a new URL

# local_zip = "kagglecatsanddogs_3367a.zip"
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall("kagglecatsanddogs")
# zip_ref.close()
#
# print(len(os.listdir('kagglecatsanddogs/PetImages/Cat/')))
# print(len(os.listdir('kagglecatsanddogs/PetImages/Dog/')))
#
# # Expected Output:
# # 12501
# # 12501
#
# # Use os.mkdir to create your directories
# # You will need a directory for cats-v-dogs, and subdirectories for training
# # and testing. These in turn will need subdirectories for 'cats' and 'dogs'
# try:
#     os.mkdir(os.path.join("kagglecatsanddogs", "training"))
#     os.mkdir(os.path.join("kagglecatsanddogs", "training", "cats"))
#     os.mkdir(os.path.join("kagglecatsanddogs", "training", "dogs"))
#     os.mkdir(os.path.join("kagglecatsanddogs", "testing"))
#     os.mkdir(os.path.join("kagglecatsanddogs", "testing", "cats"))
#     os.mkdir(os.path.join("kagglecatsanddogs", "testing", "dogs"))
#     #YOUR CODE GOES HERE
# except OSError:
#     pass
#
# # Write a python function called split_data which takes
# # a SOURCE directory containing the files
# # a TRAINING directory that a portion of the files will be copied to
# # a TESTING directory that a portion of the files will be copie to
# # a SPLIT SIZE to determine the portion
# # The files should also be randomized, so that the training set is a random
# # X% of the files, and the test set is the remaining files
# # SO, for example, if SOURCE is PetImages/Cat, and SPLIT SIZE is .9
# # Then 90% of the images in PetImages/Cat will be copied to the TRAINING dir
# # and 10% of the images will be copied to the TESTING dir
# # Also -- All images should be checked, and if they have a zero file length,
# # they will not be copied over
# #
# # os.listdir(DIRECTORY) gives you a listing of the contents of that directory
# # os.path.getsize(PATH) gives you the size of the file
# # copyfile(source, destination) copies a file from source to destination
# # random.sample(list, len(list)) shuffles a list
# def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
# # YOUR CODE STARTS HERE
#     img_list = []
#     for img in os.listdir(SOURCE):
#         if os.path.getsize(os.path.join(SOURCE, img)) > 0:
#             img_list.append(img)
#         else:
#             print(img, "is zero length, so ignoring")
#     img_list = random.sample(img_list, len(img_list))
#     training_size = int(len(img_list)*SPLIT_SIZE)
#     for i in range(len(img_list)):
#         if i < training_size:
#             copyfile(os.path.join(SOURCE, img_list[i]), os.path.join(TRAINING, img_list[i]))
#         else:
#             copyfile(os.path.join(SOURCE, img_list[i]), os.path.join(TESTING, img_list[i]))
# # YOUR CODE ENDS HERE
#
#
CAT_SOURCE_DIR = "kagglecatsanddogs/PetImages/Cat/"
TRAINING_CATS_DIR = "kagglecatsanddogs/training/cats/"
TESTING_CATS_DIR = "kagglecatsanddogs/testing/cats/"
DOG_SOURCE_DIR = "kagglecatsanddogs/PetImages/Dog/"
TRAINING_DOGS_DIR = "kagglecatsanddogs/training/dogs/"
TESTING_DOGS_DIR = "kagglecatsanddogs/testing/dogs/"
#
# split_size = .9
# split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
# split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
# # Expected output
# # 666.jpg is zero length, so ignoring
# # 11702.jpg is zero length, so ignoring
#
# print(len(os.listdir('kagglecatsanddogs/training/cats/')))
# print(len(os.listdir('kagglecatsanddogs/training/dogs/')))
# print(len(os.listdir('kagglecatsanddogs/testing/cats/')))
# print(len(os.listdir('kagglecatsanddogs/testing/dogs/')))
# Expected output:
# 11250
# 11250
# 1250
# 1250

# DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
# USE AT LEAST 3 CONVOLUTION LAYERS
model = tf.keras.models.Sequential([
# YOUR CODE HERE
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

TRAINING_DIR = "kagglecatsanddogs/training/" #YOUR CODE HERE
train_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    fill_mode='nearest'
) #YOUR CODE HERE
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    batch_size=100,
    class_mode='binary'
) #YOUR CODE HERE

VALIDATION_DIR = "kagglecatsanddogs/testing/" #YOUR CODE HERE
validation_datagen = ImageDataGenerator(rescale=1/255) #YOUR CODE HERE
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    batch_size=50,
    class_mode='binary'
) #YOUR CODE HERE

# Expected Output:
# Found 22498 images belonging to 2 classes.
# Found 2500 images belonging to 2 classes.

history = model.fit(train_generator,
                    epochs=5,
                    verbose=1,
                    validation_data=validation_generator)

# The expectation here is that the model will train, and that accuracy will be > 95% on both training and validation
# i.e. acc:A1 and val_acc:A2 will be visible, and both A1 and A2 will be > .9

# PLOT LOSS AND ACCURACY
# %matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')
# Desired output. Charts with training and validation metrics. No crash :)
