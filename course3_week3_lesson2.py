import tensorflow as tf
from tensorflow import keras
import os
# import matplotlib.pyplot as plt
import json
import numpy as np

with open(os.path.join("..", "pythonProject", "Sarcasm_Headlines_Dataset.json"), 'r') as file:
    sarcasm = json.load(file)

sentences, labels = [], []
for row in sarcasm:
    sentences.append(row['headline'])
    labels.append(row['is_sarcastic'])

training_size = 20000
num_vocab = 1000
oov_token = "<OOV>"
embd_dim = 16
num_epochs = 10
max_len = 120
trunc_type = "post"
pad_type = "post"

train_sentences = sentences[:training_size]
train_labels = labels[:training_size]
test_sentences = sentences[training_size:]
test_labels = labels[training_size:]

tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_vocab, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)
train_seq = tokenizer.texts_to_sequences(train_sentences)
train_pad = keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=max_len, truncating=trunc_type, padding=pad_type)
train_pad = np.array(train_pad)
train_labels = np.array(train_labels)
test_seq = tokenizer.texts_to_sequences(test_sentences)
test_pad = keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=max_len, truncating=trunc_type, padding=pad_type)
test_pad = np.array(test_pad)
test_labels = np.array(test_labels)

model = keras.models.Sequential([
    keras.layers.Embedding(num_vocab, embd_dim, input_length=max_len),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(24, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer=tf.optimizers.RMSprop(lr=0.0001), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
history = model.fit(train_pad, train_labels, epochs=num_epochs, validation_data=(test_pad, test_labels))
"""
Epoch 1/10
625/625 [==============================] - 10s 16ms/step - loss: 0.6825 - accuracy: 0.5598 - val_loss: 0.6647 - val_accuracy: 0.5633
Epoch 2/10
625/625 [==============================] - 10s 16ms/step - loss: 0.6125 - accuracy: 0.6723 - val_loss: 0.5494 - val_accuracy: 0.7520
Epoch 3/10
625/625 [==============================] - 31s 50ms/step - loss: 0.4918 - accuracy: 0.7705 - val_loss: 0.4752 - val_accuracy: 0.7690
Epoch 4/10
625/625 [==============================] - 9s 15ms/step - loss: 0.4353 - accuracy: 0.7997 - val_loss: 0.4286 - val_accuracy: 0.8012
Epoch 5/10
625/625 [==============================] - 9s 15ms/step - loss: 0.4056 - accuracy: 0.8149 - val_loss: 0.4108 - val_accuracy: 0.8098
Epoch 6/10
625/625 [==============================] - 10s 16ms/step - loss: 0.3870 - accuracy: 0.8231 - val_loss: 0.4056 - val_accuracy: 0.8111
Epoch 7/10
625/625 [==============================] - 10s 15ms/step - loss: 0.3751 - accuracy: 0.8290 - val_loss: 0.3912 - val_accuracy: 0.8189
Epoch 8/10
625/625 [==============================] - 9s 15ms/step - loss: 0.3661 - accuracy: 0.8328 - val_loss: 0.3901 - val_accuracy: 0.8189
Epoch 9/10
625/625 [==============================] - 9s 15ms/step - loss: 0.3584 - accuracy: 0.8379 - val_loss: 0.3892 - val_accuracy: 0.8183
Epoch 10/10
625/625 [==============================] - 10s 15ms/step - loss: 0.3534 - accuracy: 0.8393 - val_loss: 0.3865 - val_accuracy: 0.8196
"""

# def plot_graphs(history, string):
#     plt.plot(history.history[string])
#     plt.plot(history.history['val_'+string])
#     plt.xlabel("Epochs")
#     plt.ylabel(string)
#     plt.legend([string, 'val_'+string])
#     plt.show()
#
# plot_graphs(history, 'accuracy')
# plot_graphs(history, 'loss')


model = keras.models.Sequential([
    keras.layers.Embedding(num_vocab, embd_dim, input_length=max_len),
    keras.layers.Conv1D(128, 5, activation=tf.nn.relu),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer=tf.optimizers.RMSprop(lr=0.0001), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
history = model.fit(train_pad, train_labels, epochs=num_epochs, validation_data=(test_pad, test_labels))
"""
Epoch 1/10
625/625 [==============================] - 3s 4ms/step - loss: 0.6869 - accuracy: 0.5603 - val_loss: 0.6842 - val_accuracy: 0.5633
Epoch 2/10
625/625 [==============================] - 2s 4ms/step - loss: 0.6823 - accuracy: 0.5603 - val_loss: 0.6780 - val_accuracy: 0.5634
Epoch 3/10
625/625 [==============================] - 2s 4ms/step - loss: 0.6686 - accuracy: 0.5829 - val_loss: 0.6570 - val_accuracy: 0.6038
Epoch 4/10
625/625 [==============================] - 2s 4ms/step - loss: 0.6407 - accuracy: 0.6471 - val_loss: 0.6252 - val_accuracy: 0.6752
Epoch 5/10
625/625 [==============================] - 2s 4ms/step - loss: 0.6058 - accuracy: 0.6840 - val_loss: 0.5923 - val_accuracy: 0.6991
Epoch 6/10
625/625 [==============================] - 2s 4ms/step - loss: 0.5741 - accuracy: 0.7069 - val_loss: 0.5665 - val_accuracy: 0.7129
Epoch 7/10
625/625 [==============================] - 2s 4ms/step - loss: 0.5485 - accuracy: 0.7224 - val_loss: 0.5456 - val_accuracy: 0.7262
Epoch 8/10
625/625 [==============================] - 2s 4ms/step - loss: 0.5285 - accuracy: 0.7344 - val_loss: 0.5289 - val_accuracy: 0.7354
Epoch 9/10
625/625 [==============================] - 3s 4ms/step - loss: 0.5107 - accuracy: 0.7477 - val_loss: 0.5150 - val_accuracy: 0.7444
Epoch 10/10
625/625 [==============================] - 2s 4ms/step - loss: 0.4960 - accuracy: 0.7572 - val_loss: 0.5037 - val_accuracy: 0.7527
"""

# plot_graphs(history, 'accuracy')
# plot_graphs(history, 'loss')