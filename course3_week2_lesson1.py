import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import io

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
# by now the label is list, and need to convert to numpy
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

num_vocab = 10000
max_len = 120
embedding_dim = 16
tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
train_seq = tokenizer.texts_to_sequences(train_sentences)
train_pad = keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=max_len)
test_seq = tokenizer.texts_to_sequences(test_sentences)
test_pad = keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=max_len)

word_index = tokenizer.word_index
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

model = keras.models.Sequential([
    keras.layers.Embedding(num_vocab, embedding_dim, input_length=120),
    keras.layers.Flatten(),
    # keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(6, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)  # classify if sentence is sarcastic
])
model.compile(optimizer=tf.optimizers.RMSprop(lr=0.0001), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
model.fit(train_pad, train_labels, epochs=30, validation_data=(test_pad, test_labels))
"""
Epoch 28/30
782/782 [==============================] - 2s 2ms/step - loss: 0.2644 - accuracy: 0.8962 - val_loss: 0.3149 - val_accuracy: 0.8643
Epoch 29/30
782/782 [==============================] - 2s 2ms/step - loss: 0.2612 - accuracy: 0.8972 - val_loss: 0.3138 - val_accuracy: 0.8651
Epoch 30/30
782/782 [==============================] - 2s 2ms/step - loss: 0.2582 - accuracy: 0.8978 - val_loss: 0.3131 - val_accuracy: 0.8651
"""

test = ["It's so good you won't believe it"]
test_pad = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(test), maxlen=max_len)
print(model.predict(test_pad))  # [[0.698234]]

embd = model.layers[0]
embd_weights = embd.get_weights()[0]
print(embd_weights.shape)  # (10000, 16)

out_v = io.open('course3_week2_lesson1_vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('course3_week2_lesson1_meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, num_vocab):
    word = reverse_word_index[word_num]
    embeddings = embd_weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()