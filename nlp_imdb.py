import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import io

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, GlobalAveragePooling1D, Bidirectional, LSTM, GRU, Conv1D

# load IMDB dataset in tensorflow-datasets
print(tf.__version__)
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
# items in train_data and test_data are Tensors
train_data, test_data = imdb['train'], imdb['test']

# split sentences and labels in training and testing dataset
training_sentences, training_labels = [], []
testing_sentences, testing_labels = [], []
# convert sentence and label in Tensors in train_data and test_data to numpy object
for sent, label in train_data:
    training_sentences.append(sent.numpy().decode('utf8'))
    training_labels.append(label.numpy())
for sent, label in test_data:
    testing_sentences.append(sent.numpy().decode('utf8'))
    testing_labels.append(label.numpy())

training_sentences = np.array(training_sentences)
training_labels = np.array(training_labels)
testing_sentences = np.array(testing_sentences)
testing_labels = np.array(testing_labels)
print(type(training_sentences[0]), type(training_labels[0]))
print(training_sentences[0], training_labels[0])
print(training_sentences.shape, training_labels.shape)

# convert text to padded numerical sequence using Tokenizer and pad_sequences
vocab_size = 10000
oov_tok = "<OOV>"
max_len = 120
pad_type = 'post'
trunc_type = 'post'
embd_dim = 16

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
words_index = tokenizer.word_index
print(len(words_index))
training_seq = tokenizer.texts_to_sequences(training_sentences)
training_pad = pad_sequences(training_seq, maxlen=max_len, padding=pad_type, truncating=trunc_type)
testing_seq = tokenizer.texts_to_sequences(testing_sentences)
testing_pad = pad_sequences(testing_seq, maxlen=max_len, padding=pad_type, truncating=trunc_type)

# define model and fit with data
model = Sequential()
model.add(Embedding(vocab_size, embd_dim, input_shape=(max_len,)))
# model.add(Flatten())
model.add(GlobalAveragePooling1D())
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 120, 16)           160000    
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0         
_________________________________________________________________
dense (Dense)                (None, 6)                 102       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 7         
=================================================================
Total params: 160,109
Trainable params: 160,109
Non-trainable params: 0
"""
# model.fit(training_pad, training_labels, batch_size=10, epochs=3, validation_data=(testing_pad, testing_labels))
#
# # output TSV file of vectors and metadata for projector.tensorflow.org
# embd = model.layers[0]
# weights = embd.get_weights()[0]
# print(weights.shape)  # shape = (vocab_size, embd_dim)
#
# reverse_words_index = dict([(value, key) for (key, value) in words_index.items()])
# out_v = io.open("vecs.tsv", 'w', encoding='utf-8')
# out_m = io.open("meta.tsv", 'w', encoding='utf-8')
# for word_num in range(1, vocab_size):
#     word = reverse_words_index[word_num]
#     embeddings = weights[word_num]
#     out_m.write(word + "\n")
#     out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
# out_v.close()
# out_m.close()


## try IMDB's subwords8k dataset with various models
# bidirectional LSTM
# dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
# define vanilla model and fit with data
model = Sequential()
model.add(Embedding(vocab_size, embd_dim, input_shape=(max_len,)))
model.add(GlobalAveragePooling1D())
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# history = model.fit(training_pad, training_labels, batch_size=10, epochs=3, validation_data=(testing_pad, testing_labels))
"""
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 120, 16)           160000    
_________________________________________________________________
global_average_pooling1d_1 ( (None, 16)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 24)                408       
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 25        
=================================================================
Total params: 160,433
Trainable params: 160,433
Non-trainable params: 0
"""
# define LSTM model and fit with data
model = Sequential()
model.add(Embedding(vocab_size, embd_dim, input_shape=(max_len,)))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# history = model.fit(training_pad, training_labels, batch_size=10, epochs=3, validation_data=(testing_pad, testing_labels))
"""
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, 120, 16)           160000    
_________________________________________________________________
bidirectional (Bidirectional (None, 64)                12544     
_________________________________________________________________
dense_4 (Dense)              (None, 24)                1560      
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 25        
=================================================================
Total params: 174,129
Trainable params: 174,129
Non-trainable params: 0
"""
# define GRU model and fit with data
model = Sequential()
model.add(Embedding(vocab_size, embd_dim, input_shape=(max_len,)))
model.add(Bidirectional(GRU(32)))
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# history = model.fit(training_pad, training_labels, batch_size=10, epochs=3, validation_data=(testing_pad, testing_labels))
"""
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, 120, 16)           160000    
_________________________________________________________________
bidirectional_1 (Bidirection (None, 64)                9600      
_________________________________________________________________
dense_6 (Dense)              (None, 24)                1560      
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 25        
=================================================================
Total params: 171,185
Trainable params: 171,185
Non-trainable params: 0
"""
# define convolution model and fit with data
model = Sequential()
model.add(Embedding(vocab_size, embd_dim, input_shape=(max_len,)))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(24, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# history = model.fit(training_pad, training_labels, batch_size=10, epochs=3, validation_data=(testing_pad, testing_labels))
"""
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_4 (Embedding)      (None, 120, 16)           160000    
_________________________________________________________________
conv1d (Conv1D)              (None, 116, 128)          10368     
_________________________________________________________________
global_average_pooling1d_2 ( (None, 128)               0         
_________________________________________________________________
dense_8 (Dense)              (None, 24)                3096      
_________________________________________________________________
dense_9 (Dense)              (None, 1)                 25        
=================================================================
Total params: 173,489
Trainable params: 173,489
Non-trainable params: 0
_________________________________________________________________

Process finished with exit code 0

"""