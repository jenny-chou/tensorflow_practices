import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

# practice with Kaggle data
# initialize parameters
training_size = 20000
vocab_size = 10000
oov_token = "<OOV>"
max_length = 100
padding_type = 'post'
trunc_type = 'post'
embedding_dim = 16
num_epochs = 5
batch_size = 10

# read raw data from json
with open("Sarcasm_Headlines_Dataset.json", 'r') as file:
    datastore = json.load(file)

# extract json
sentences, labels, urls = [], [], []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# split train and test set
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
print(training_sentences[0])

# neural net only sees training data. So tokenizer should fit to only training data
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)

# word_index is dict of all the unique words and its index value
# length of dict is the number of unique tokens/words
word_index = tokenizer.word_index
print(len(word_index))

# convert training set to numerical sequence
training_sequences = tokenizer.texts_to_sequences(training_sentences)

# transform training sequence to same length by padding or truncating
training_padded = pad_sequences(training_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)
print("training_sequences[0]:", training_sequences[0])
print("training_padded[0]:", training_padded[0])
print("training_padded.shape", training_padded.shape)

# convert and transform testing set
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

# convert to array to get it to work with TensorFlow 2.x
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# create and compile embedded model
model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(1, activation='sigmoid')])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
"""
model.summary()

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 100, 16)           160000
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0
_________________________________________________________________
dense (Dense)                (None, 24)                408
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 25
=================================================================
Total params: 160,433
Trainable params: 160,433
Non-trainable params: 0
"""

# fit model with training set and validate with testing set
history = model.fit(training_padded, training_labels, batch_size=batch_size, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels))
print(history.history['val_accuracy'])
print(history.history['accuracy'])

# # plot accuracy and loss
# def plot_graphs(history, string):
#     plt.plot(history.history[string])
#     plt.plot(history.history['val'+string])
#     plt.xlabel("epochs")
#     plt.ylabel(string)
#     plt.legand([string, 'val_'+string])
#     plt.show()
#
# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")

# # predict new sentences if they're sarcastic
# seed = [
#     "granny starting to fear spiders in the garden might be real",
#     "the weather today is bright and sunny"
# ]
# seed_seq = tokenizer.texts_to_sequences(seed)
# seed_pad = pad_sequences(seed_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# print(model.predict(seed_pad))


# # use bidirectional LSTM layers to make prediction
# binary classification problem: Is this sentence sarcastic? or not sarcastic?
embedding_dim = 64
model2 = Sequential([
    Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    # wrap LSTM in Bidirectional:
    # Looks at sentence forward & backward
    # Learn best parameter in each directions and merge them
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
"""
model2.summary()

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, None, 64)          640000    
_________________________________________________________________
bidirectional (Bidirectional (None, None, 128)         66048     
_________________________________________________________________
bidirectional_1 (Bidirection (None, 128)               98816     
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 65        
=================================================================
Total params: 813,185
Trainable params: 813,185
Non-trainable params: 0
"""
# model2.fit(training_padded, training_labels, batch_size=batch_size, epochs=num_epochs,
#            validation_data=(testing_padded, testing_labels))
#
# # predict new sentences if they're sarcastic
# seed = [
#     "granny starting to fear spiders in the garden might be real",
#     "the weather today is bright and sunny"
# ]
# seed_seq = tokenizer.texts_to_sequences(seed)
# seed_pad = pad_sequences(seed_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# print(model2.predict(seed_pad))
"""
[[5.829891e-01]
 [6.624551e-05]]
"""