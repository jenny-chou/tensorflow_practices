import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(
    input_shape=(150,150,3),
    include_top=False,
    weights=None
) # Your Code Here

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable=False # Your Code Here

# Print the model summary
pre_trained_model.summary()

last_layer = pre_trained_model.get_layer("mixed7")  # Your Code Here)
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output # Your Code Here
# Expected Output:
# ('last layer output shape: ', (None, 7, 7, 768))


# Define a Callback class that stops training once accuracy reaches 99.9%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(256, activation='relu')(x) # Your Code Here)(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x) # Your Code Here)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x) # Your Code Here)(x)

model = Model(inputs=pre_trained_model.input, outputs=x)  # Your Code Here, x)

model.compile(optimizer = RMSprop(lr=0.0001),
              loss = 'binary_crossentropy', # Your Code Here,
              metrics = ['accuracy']) # Your Code Here)

# model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_horses_dir = "horse-or-human/horses"# Your Code Here
train_humans_dir = "horse-or-human/humans"# Your Code Here
validation_horses_dir = "validation-horse-or-human/horses"# Your Code Here
validation_humans_dir = "validation-horse-or-human/humans"# Your Code Here

train_horses_fnames = os.listdir(train_horses_dir)# Your Code Here
train_humans_fnames = os.listdir(train_humans_dir)# Your Code Here
validation_horses_fnames = os.listdir(validation_horses_dir)# Your Code Here
validation_humans_fnames = os.listdir(validation_humans_dir)# Your Code Here

print(len(train_horses_fnames))# Your Code Here)
print(len(train_humans_fnames))# Your Code Here)
print(len(validation_horses_fnames))# Your Code Here)
print(len(validation_humans_fnames))# Your Code Here)

# Expected Output:
# 500
# 527
# 128
# 128

# Define our example directories and files
train_dir = "horse-or-human"
validation_dir = "validation-horse-or-human"

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    fill_mode='nearest'
)    # Your Code Here)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(
    train_dir,
    target_size=(150,150),
    batch_size=8,
    class_mode='binary'
)# Your Code Here )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(rescale=1/255) # Your Code Here)

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=8,
    class_mode='binary'
)# Your Code Here)

# Expected Output:
# Found 1027 images belonging to 2 classes.
# Found 256 images belonging to 2 classes.


# Run this and see how many epochs it should take before the callback
# fires, and stops training at 99.9% accuracy
# (It should take less than 100 epochs)

callbacks = myCallback() # Your Code Here
history = model.fit(train_generator, epochs=10, validation_data=validation_generator, callbacks=[callbacks]) # Your Code Here)


import matplotlib.pyplot as plt
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