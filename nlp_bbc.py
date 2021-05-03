import tensorflow as tf
import csv
import io
import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

vocab_size = 1000
embedding_dim = 16
max_len = 120
trunc_type = 'post'
padding_type = 'post'
oov_token = "<OOV>"
training_portion = 0.8
epochs = 30

stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
              "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
              "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
              "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
              "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its",
              "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
              "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
              "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
              "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
              "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've",
              "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
              "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
              "yourself", "yourselves" ]
sentences, labels = [], []
with open(os.path.join("..", "TFExams", "bbc-text.csv"), 'r') as file:
    bbc = csv.reader(file, delimiter=',')
    next(bbc)
    for row in bbc:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            sentence = sentence.replace(" "+word+" ", " ")
            sentence = sentence.replace("  ", " ")
        sentences.append(sentence)
print(len(labels))  # 2225
print(len(sentences))  # 2225
print(sentences[0])  # tv future hands viewers home theatre systems plasma high-definition...

train_size = int(len(sentences)*training_portion)
train_sentences = sentences[:train_size]
train_labels = labels[:train_size]
test_sentences = sentences[train_size:]
test_labels = labels[train_size:]
print(train_size)  # 1780
print(len(train_sentences))  # 1780
print(len(train_labels))  # 1780
print(len(test_sentences))  #445
print(len(test_labels))  # 445

tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_len, truncating=trunc_type, padding=padding_type)
print(len(train_sequences[0]))  #449
print(len(train_padded[0]))  # 120
print(len(train_sequences[1]))  # 200
print(len(train_padded[1]))  # 120
print(len(train_sequences[10]))  # 192
print(len(train_padded[10]))  # 120

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_len, truncating=trunc_type, padding=padding_type)
print(len(test_sequences))  # 445
print(test_padded.shape)  # (445, 120)

label_tokenizer = keras.preprocessing.text.Tokenizer()
label_tokenizer.fit_on_texts(labels)
train_label_seq = label_tokenizer.texts_to_sequences(train_labels)
test_label_seq = label_tokenizer.texts_to_sequences(test_labels)
train_label_seq, test_label_seq = np.array(train_label_seq), np.array(test_label_seq)
print(train_label_seq[0])  # [4]
print(train_label_seq[1])  # [2]
print(train_label_seq[2])  # [1]
print(train_label_seq.shape)  # (1780, 1)
print(test_label_seq[0])  # [5]
print(test_label_seq[1])  # [4]
print(test_label_seq[2])  # [3]
print(test_label_seq.shape)  # (445, 1)
print(label_tokenizer.word_index)  # {'sport': 1, 'business': 2, 'politics': 3, 'tech': 4, 'entertainment': 5}

model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(len(label_tokenizer.word_index)+1, activation='softmax')
    # label token have total 5 classes and starts from 1, so categories must have range of [0, 6) to include all classes
])
model.compile(optimizer=tf.optimizers.RMSprop(lr=0.001), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.summary()
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 120, 16)           16000     
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0         
_________________________________________________________________
dense (Dense)                (None, 24)                408       
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 150       
=================================================================
Total params: 16,558
Trainable params: 16,558
Non-trainable params: 0
_________________________________________________________________
"""
history = model.fit(train_padded, train_label_seq, epochs=epochs, validation_data=(test_padded, test_label_seq))
"""
Epoch 9/10
56/56 [==============================] - 0s 1ms/step - loss: 0.8810 - accuracy: 0.7938 - val_loss: 0.8516 - val_accuracy: 0.8337
Epoch 10/10
56/56 [==============================] - 0s 1ms/step - loss: 0.7612 - accuracy: 0.8663 - val_loss: 0.7412 - val_accuracy: 0.8517
"""

plt.plot(range(epochs), history.history['accuracy'], 'r', label="Training accuracy")
plt.plot(range(epochs), history.history['val_accuracy'], 'b', label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()

plt.plot(range(epochs), history.history['loss'], 'r', label="Training loss")
plt.plot(range(epochs), history.history['val_loss'], 'b', label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

embd = model.layers[0]
weights = embd.get_weights()[0]
print(weights.shape)  # (1000, 16)

reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])
out_v = io.open('nlp_bbc_vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('nlp_bbc_meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()