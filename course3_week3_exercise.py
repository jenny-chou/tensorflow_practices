import tensorflow as tf
import numpy as np
from tensorflow import keras
import csv
import random
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

corpus = []
num_sentences = 0
with open("training_cleaned.csv", 'r', encoding='utf8', errors='ignore') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        item = [row[5], 1 if row[0] == 4 else 0]
        num_sentences += 1
        corpus.append(item)
print(num_sentences)  # 1600000
print(len(corpus))  # 1600000
print(corpus[1])  # ["is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!", 0]

embedding_dim = 100
max_length = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size= 160000
test_portion=.1
epochs=5

random.shuffle(corpus)
sentences, labels = [], []
for item in corpus[:training_size]:
    sentences.append(item[0])
    labels.append(item[1])

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
vocab_size = len(word_index)
print(vocab_size)  # 138622
print(word_index['i'])  # 1

sequences = tokenizer.texts_to_sequences(sentences)
padded = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type)
split = int(test_portion * training_size)
test_sequences = np.array(padded[:split])
training_sequences = np.array(padded[split:])
test_labels = np.array(labels[:split])
training_labels = np.array(labels[split:])


model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
    keras.layers.Conv1D(embedding_dim*8, 5, activation=tf.nn.relu),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
model.fit(training_sequences, training_labels, epochs=epochs, validation_data=(test_sequences, test_labels))
"""
Epoch 1/5
4500/4500 [==============================] - 512s 114ms/step - loss: 0.0012 - accuracy: 1.0000 - val_loss: 1.9667e-07 - val_accuracy: 1.0000
Epoch 2/5
4500/4500 [==============================] - 509s 113ms/step - loss: 4.1988e-09 - accuracy: 1.0000 - val_loss: 1.6754e-07 - val_accuracy: 1.0000
Epoch 3/5
4500/4500 [==============================] - 506s 112ms/step - loss: 2.0856e-09 - accuracy: 1.0000 - val_loss: 1.1290e-07 - val_accuracy: 1.0000
Epoch 4/5
4500/4500 [==============================] - 514s 114ms/step - loss: 6.4344e-10 - accuracy: 1.0000 - val_loss: 6.5346e-08 - val_accuracy: 1.0000
Epoch 5/5
4500/4500 [==============================] - 549s 122ms/step - loss: 1.7806e-10 - accuracy: 1.0000 - val_loss: 3.7595e-08 - val_accuracy: 1.0000
"""