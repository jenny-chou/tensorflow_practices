import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

corpus = open("sonnets.txt").read()
corpus = corpus.lower().split('\n')

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index)+1

sentences = []
for row in corpus:
    sentence = tokenizer.texts_to_sequences([row])[0]
    for i in range(len(sentence)):
        sentences.append(sentence[:i+1])
sentences = np.array(sentences)
max_len = max([len(row) for row in sentences])
sentences = keras.preprocessing.sequence.pad_sequences(sentences, maxlen=max_len)
print(np.array(sentences).shape)  # (17618, 11)

xs, ys = sentences[:, :-1], sentences[:, -1]
ys = keras.utils.to_categorical(ys, num_classes=total_words)

model = keras.models.Sequential([
    keras.layers.Embedding(total_words, 64, input_length=max_len-1),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(total_words, activation='softmax')
])
model.compile(optimizer=tf.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(xs, ys, epochs=100)

reverse_word_index = dict([(index, word) for (word, index) in tokenizer.word_index.items()])
seed = "When thou art old I take thee flowers to thy grave"
for _ in range(200):
    seed_pad = tokenizer.texts_to_sequences([seed])[0]
    seed_pad = keras.preprocessing.sequence.pad_sequences([seed_pad], maxlen=max_len-1)
    predict = model.predict(seed_pad)
    predict_word = reverse_word_index[np.argmax(predict)]
    seed += " "+predict_word
print(seed)
"""
Epoch 97/100
551/551 [==============================] - 4s 7ms/step - loss: 1.1470 - accuracy: 0.7436
Epoch 98/100
551/551 [==============================] - 4s 7ms/step - loss: 1.1658 - accuracy: 0.7400
Epoch 99/100
551/551 [==============================] - 4s 7ms/step - loss: 1.1532 - accuracy: 0.7424
Epoch 100/100
551/551 [==============================] - 4s 7ms/step - loss: 1.1311 - accuracy: 0.7487
When thou art old I take thee flowers to thy grave staineth burn ' groan me torn more in me 
am can pleasure that die thou love thy spring still level of his rest heart room room room 
great day burn grew treasure die not in his wretch's fall and store not waste from still show 
me or dead can be true lies hate me torn torn me burn time laid that hate still it took sun 
bow despise despise state ' ' of love or live by yet we have live by thee is in love me more 
love thee more blessed in die time heart cruel place is in me thou sin pay thy great heart 
lover room date mark at place thou phrase she by me fair art perjured beauty as love more all 
lie in me me be a painted war in me not so dear heart repair great verse die hate in me 
hell ' love me for love though hate ' me forth thee burn me at heart me knowst thou verse that 
love dear heart in love 'no ' ' his tiger's waste day day day and laid live in thee i live as 
love so more remedy heinous every looks ' one part

"""