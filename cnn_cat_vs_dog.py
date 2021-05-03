import os
import zipfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras

"""
zip_file_path = "cats_and_dogs_filtered.zip"
zip_ref = zipfile.ZipFile(zip_file_path, 'r')
zip_ref.extractall(base_dir)
zip_ref.close()
"""

base_dir = "..\TFExams\cats_and_dogs\cats_and_dogs_filtered"
train_dir = os.path.join(base_dir, "train")
train_dog_dir = os.path.join(train_dir, "dogs")
train_cat_dir = os.path.join(train_dir, "cats")
test_dir = os.path.join(base_dir, "validation")
test_dog_dir = os.path.join(test_dir, "dogs")
test_cat_dir = os.path.join(test_dir, "cats")

print("Total train dog and cat images:", len(os.listdir(train_dog_dir)), len(os.listdir(train_cat_dir)))
print("Total test dog and cat images:", len(os.listdir(test_dog_dir)), len(os.listdir(test_cat_dir)))

num_rows, num_cols = 4, 4
pic_index = 0
fig = plt.gcf()
fig.set_size_inches(num_cols*4, num_rows*4)
pic_index += 8
next_cat_pic = [os.path.join(train_cat_dir, fname) for fname in os.listdir(train_cat_dir)[pic_index-8 : pic_index]]
next_dog_pic = [os.path.join(train_dog_dir, fname) for fname in os.listdir(train_dog_dir)[pic_index-8 : pic_index]]

# for i, img_path in enumerate(next_cat_pic + next_dog_pic):
#     sp = plt.subplot(num_rows, num_cols, i+1)
#     sp.axis('Off')
#     img = mpimg.imread(img_path)
#     plt.imshow(img)
# plt.figure()

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
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
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu, input_shape=(150,150,3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer=tf.optimizers.RMSprop(lr=0.0001), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
epochs = 100
history = model.fit(train_generator, steps_per_epoch=100, epochs=epochs, validation_data=test_generator, validation_steps=50)

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

model_layers_outputs = [layer.output for layer in model.layers]
model_layers_names = [layer.name for layer in model.layers]
model_input = model.input
model_Model = keras.models.Model(inputs=model_input, outputs=model_layers_outputs)

cat_img_path = [os.path.join(train_cat_dir, fname) for fname in os.listdir(train_cat_dir)]
dog_img_path = [os.path.join(train_dog_dir, fname) for fname in os.listdir(train_dog_dir)]
imgs_path = np.random.choice(cat_img_path + dog_img_path)
imgs = keras.preprocessing.image.load_img(imgs_path, target_size=(150,150))
imgs_arr = keras.preprocessing.image.img_to_array(imgs)
imgs_arr = imgs_arr.reshape((1,150,150,3))/255
predictions = model_Model.predict(imgs_arr)

for layer in range(8):
    feature_maps = predictions[layer]
    num_features = feature_maps.shape[3]
    size_features = feature_maps.shape[1]
    display_grid = np.zeros((size_features, size_features * num_features))
    for i in range(num_features):
        feature_map = feature_maps[0,:,:,i]
        feature_map -= feature_map.mean()
        if(feature_map.std()==0):
            feature_map /= 0.0001
        else:
            feature_map /= feature_map.std()
        feature_map *= 64
        feature_map += 128
        feature_map = np.clip(feature_map, 0, 255).astype('uint8')
        display_grid[:, i * size_features:(i + 1) * size_features] = feature_map
    scale = 20. / num_features
    plt.figure(figsize=(scale * num_features, scale))
    plt.title(model_layers_names[layer])
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()
