import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import os
# import matplotlib.pyplot as plt
import numpy as np

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

train_sentences, train_labels = [], []
test_sentences, test_labels = [], []
for sent, label in train_data:
    train_sentences.append(sent.numpy().decode('utf8'))
    train_labels.append(label.numpy())
for sent, label in test_data:
    test_sentences.append(sent.numpy().decode('utf8'))
    test_labels.append(label.numpy())
train_sentences, train_labels = np.array(train_sentences), np.array(train_labels)
test_sentences, test_labels = np.array(test_sentences), np.array(test_labels)

num_vocab = 1000
oov_token = "<OOV>"
max_len = 120
trunc_type = "post"
pad_type = "post"
embd_dim = 16
epochs = 10

tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_vocab, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)
train_seq = tokenizer.texts_to_sequences(train_sentences)
train_pad = keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=max_len, truncating=trunc_type, padding=pad_type)
test_seq = tokenizer.texts_to_sequences(test_sentences)
test_pad = keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=max_len, truncating=trunc_type, padding=pad_type)

model = keras.models.Sequential([
    keras.layers.Embedding(num_vocab, embd_dim, input_length=max_len),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(24, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer=tf.optimizers.RMSprop(lr=0.0001), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
model.summary()
history = model.fit(train_pad, train_labels, epochs=epochs, validation_data=(test_pad, test_labels))
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 120, 16)           16000     
_________________________________________________________________
bidirectional (Bidirectional (None, 128)               41472     
_________________________________________________________________
dense (Dense)                (None, 24)                3096      
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 25        
=================================================================
Total params: 60,593
Trainable params: 60,593
Non-trainable params: 0
_________________________________________________________________

Epoch 1/10
782/782 [==============================] - 18s 22ms/step - loss: 0.6779 - accuracy: 0.5657 - val_loss: 0.6335 - val_accuracy: 0.6462
Epoch 2/10
782/782 [==============================] - 16s 20ms/step - loss: 0.5530 - accuracy: 0.7185 - val_loss: 0.4767 - val_accuracy: 0.7784
Epoch 3/10
782/782 [==============================] - 16s 20ms/step - loss: 0.4610 - accuracy: 0.7864 - val_loss: 0.4408 - val_accuracy: 0.7986
Epoch 4/10
782/782 [==============================] - 16s 20ms/step - loss: 0.4295 - accuracy: 0.8053 - val_loss: 0.4290 - val_accuracy: 0.7963
Epoch 5/10
782/782 [==============================] - 16s 21ms/step - loss: 0.4149 - accuracy: 0.8139 - val_loss: 0.4384 - val_accuracy: 0.8002
Epoch 6/10
782/782 [==============================] - 16s 20ms/step - loss: 0.4053 - accuracy: 0.8186 - val_loss: 0.4091 - val_accuracy: 0.8126
Epoch 7/10
782/782 [==============================] - 16s 21ms/step - loss: 0.3976 - accuracy: 0.8230 - val_loss: 0.4377 - val_accuracy: 0.7967
Epoch 8/10
782/782 [==============================] - 16s 21ms/step - loss: 0.3937 - accuracy: 0.8247 - val_loss: 0.4022 - val_accuracy: 0.8130
Epoch 9/10
782/782 [==============================] - 16s 20ms/step - loss: 0.3919 - accuracy: 0.8246 - val_loss: 0.4161 - val_accuracy: 0.8108
Epoch 10/10
782/782 [==============================] - 16s 20ms/step - loss: 0.3891 - accuracy: 0.8270 - val_loss: 0.4049 - val_accuracy: 0.8144
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
    keras.layers.Bidirectional(keras.layers.GRU(64)),
    keras.layers.Dense(24, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer=tf.optimizers.RMSprop(lr=0.0001), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
model.summary()
history = model.fit(train_pad, train_labels, epochs=epochs, validation_data=(test_pad, test_labels))
"""
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 120, 16)           16000     
_________________________________________________________________
bidirectional_1 (Bidirection (None, 128)               31488     
_________________________________________________________________
dense_2 (Dense)              (None, 24)                3096      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 25        
=================================================================
Total params: 50,609
Trainable params: 50,609
Non-trainable params: 0
_________________________________________________________________

Epoch 1/10
782/782 [==============================] - 16s 20ms/step - loss: 0.6919 - accuracy: 0.5324 - val_loss: 0.6892 - val_accuracy: 0.5633
Epoch 2/10
782/782 [==============================] - 15s 20ms/step - loss: 0.6305 - accuracy: 0.6418 - val_loss: 0.5256 - val_accuracy: 0.7360
Epoch 3/10
782/782 [==============================] - 15s 19ms/step - loss: 0.4708 - accuracy: 0.7763 - val_loss: 0.5531 - val_accuracy: 0.7174
Epoch 4/10
782/782 [==============================] - 15s 19ms/step - loss: 0.4253 - accuracy: 0.8051 - val_loss: 0.4350 - val_accuracy: 0.7972
Epoch 5/10
782/782 [==============================] - 15s 19ms/step - loss: 0.4075 - accuracy: 0.8147 - val_loss: 0.4208 - val_accuracy: 0.8012
Epoch 6/10
782/782 [==============================] - 15s 20ms/step - loss: 0.3985 - accuracy: 0.8198 - val_loss: 0.4146 - val_accuracy: 0.8063
Epoch 7/10
782/782 [==============================] - 15s 19ms/step - loss: 0.3939 - accuracy: 0.8248 - val_loss: 0.4205 - val_accuracy: 0.8058
Epoch 8/10
782/782 [==============================] - 15s 19ms/step - loss: 0.3921 - accuracy: 0.8252 - val_loss: 0.4054 - val_accuracy: 0.8121
Epoch 9/10
782/782 [==============================] - 15s 19ms/step - loss: 0.3898 - accuracy: 0.8256 - val_loss: 0.4040 - val_accuracy: 0.8129
Epoch 10/10
782/782 [==============================] - 15s 19ms/step - loss: 0.3887 - accuracy: 0.8269 - val_loss: 0.4133 - val_accuracy: 0.8068
"""

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
model.summary()
history = model.fit(train_pad, train_labels, epochs=epochs, validation_data=(test_pad, test_labels))
"""
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, 120, 16)           16000     
_________________________________________________________________
conv1d (Conv1D)              (None, 116, 128)          10368     
_________________________________________________________________
global_average_pooling1d (Gl (None, 128)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 24)                3096      
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 25        
=================================================================
Total params: 29,489
Trainable params: 29,489
Non-trainable params: 0
_________________________________________________________________

Epoch 1/10
782/782 [==============================] - 5s 6ms/step - loss: 0.6873 - accuracy: 0.6038 - val_loss: 0.6729 - val_accuracy: 0.6558
Epoch 2/10
782/782 [==============================] - 5s 6ms/step - loss: 0.6272 - accuracy: 0.6978 - val_loss: 0.5759 - val_accuracy: 0.7311
Epoch 3/10
782/782 [==============================] - 5s 6ms/step - loss: 0.5173 - accuracy: 0.7662 - val_loss: 0.4792 - val_accuracy: 0.7833
Epoch 4/10
782/782 [==============================] - 5s 6ms/step - loss: 0.4511 - accuracy: 0.7949 - val_loss: 0.4425 - val_accuracy: 0.7953
Epoch 5/10
782/782 [==============================] - 5s 6ms/step - loss: 0.4258 - accuracy: 0.8061 - val_loss: 0.4291 - val_accuracy: 0.8013
Epoch 6/10
782/782 [==============================] - 5s 6ms/step - loss: 0.4134 - accuracy: 0.8118 - val_loss: 0.4208 - val_accuracy: 0.8058
Epoch 7/10
782/782 [==============================] - 5s 6ms/step - loss: 0.4054 - accuracy: 0.8176 - val_loss: 0.4183 - val_accuracy: 0.8061
Epoch 8/10
782/782 [==============================] - 5s 6ms/step - loss: 0.4003 - accuracy: 0.8213 - val_loss: 0.4147 - val_accuracy: 0.8084
Epoch 9/10
782/782 [==============================] - 5s 6ms/step - loss: 0.3964 - accuracy: 0.8216 - val_loss: 0.4120 - val_accuracy: 0.8110
Epoch 10/10
782/782 [==============================] - 5s 7ms/step - loss: 0.3939 - accuracy: 0.8239 - val_loss: 0.4151 - val_accuracy: 0.8071
"""

# plot_graphs(history, 'accuracy')
# plot_graphs(history, 'loss')
