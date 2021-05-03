import tensorflow as tf
import numpy as np
import os
import zipfile
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras

"""
zip_file_path = "horse-or-human.zip"
zip_ref = zipfile.ZipFile(zip_file_path, 'r')
zip_ref.extractall("horse-or-human/")
zip_ref.close()

zip_file_path = "validation-horse-or-human.zip"
zip_ref = zipfile.ZipFile(zip_file_path, 'r')
zip_ref.extractall("validation-horse-or-human/")
zip_ref.close()
"""

train_horse_dir = os.path.join("horse-or-human", "horses")
train_human_dir = os.path.join("horse-or-human", "humans")
test_horse_dir = os.path.join("validation-horse-or-human", "horses")
test_human_dir = os.path.join("validation-horse-or-human", "humans")

print("total train horses and humans:", len(os.listdir(train_horse_dir)), len(os.listdir(train_human_dir)))
print("total test horses and humans:", len(os.listdir(test_horse_dir)), len(os.listdir(test_human_dir)))

num_rows, num_cols = 4, 4
pic_index = 0
fig = plt.gcf()
fig.set_size_inches(num_cols*4, num_rows*4)
pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in os.listdir(train_horse_dir)[pic_index-8 : pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) for fname in os.listdir(train_human_dir)[pic_index-8 : pic_index]]

# for i, img_path in enumerate(next_horse_pix + next_human_pix):
#     sp = plt.subplot(num_rows, num_cols, i+1)
#     sp.axis('Off')
#     img = mpimg.imread(img_path)
#     plt.imshow(img)
# plt.show()

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    "horse-or-human",
    target_size=(150,150),
    batch_size=128,
    class_mode='binary'
)
# Found 1027 images belonging to 2 classes.

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
    "validation-horse-or-human",
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)
# Found 256 images belonging to 2 classes.

model = keras.models.Sequential([
    keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, input_shape=(150, 150, 3)),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer=tf.optimizers.RMSprop(lr=0.001), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
epochs = 100
history = model.fit(train_generator, steps_per_epoch=8, epochs=epochs, validation_data=(test_generator), validation_steps=8)

# plot accuracy/loss trend
plt.plot(range(epochs), history.history['accuracy'])
plt.plot(range(epochs), history.history['val_accuracy'])
plt.title("Training and validation accuracy")
plt.legend(['accuracy', 'val_accuracy'])
plt.figure()

plt.plot(range(epochs), history.history['loss'])
plt.plot(range(epochs), history.history['val_loss'])
plt.title("Training and validation loss")
plt.legend(['loss', 'val_loss'])
plt.figure()

# plot activation maps
model_layers_names = [layer.name for layer in model.layers]
model_layers_outputs = [layer.output for layer in model.layers]
model_inputs = model.input
activation_model = keras.models.Model(inputs=model_inputs, outputs=model_layers_outputs)

horse_imgs_path = [os.path.join(train_horse_dir, fname) for fname in os.listdir(train_horse_dir)]
human_imgs_path = [os.path.join(train_human_dir, fname) for fname in os.listdir(train_human_dir)]
imgs_path = random.choice(horse_imgs_path + human_imgs_path)
imgs = tf.keras.preprocessing.image.load_img(imgs_path, target_size=(150,150))
imgs_arr = tf.keras.preprocessing.image.img_to_array(imgs)
print(imgs_arr.shape)  # (150, 150, 3)
# imgs_arr = imgs_arr.reshape((1,) + imgs_arr.shape)/255
imgs_arr = imgs_arr.reshape(1,150,150,3)/255
print(imgs_arr.shape)  # (1, 150, 150, 3)

activation_maps = activation_model.predict(imgs_arr)

for layer in range(0,6):
    num_features = activation_maps[layer].shape[3]
    width_feature = activation_maps[layer].shape[1]
    display_grid = np.zeros((width_feature, width_feature*num_features))
    for i in range(num_features):
        activation_map = activation_maps[layer][0,:,:,i]
        activation_map -= activation_map.mean()
        if(activation_map.std()==0):
            activation_map /= 0.0001
        else:
            activation_map /= activation_map.std()
        activation_map *= 64
        activation_map += 128
        activation_map = np.clip(activation_map, 0, 255).astype('uint8')
        display_grid[:, i*width_feature:(i+1)*width_feature] = activation_map
    scale = 20./num_features
    plt.figure(figsize=(scale*num_features, scale))
    plt.title(model_layers_names[layer])
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()