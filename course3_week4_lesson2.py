import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

corpus = open(os.path.join("..", "pythonProject", "Irish_lyrics.txt")).read()
corpus = corpus.lower().split('\n')

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index)+1
print(tokenizer.word_index)  # {'the': 1, 'and': 2, 'i': 3, 'to': 4, 'a': 5, 'of': 6, 'my': 7, 'in': 8, 'me': 9 ...}
print(total_words)  # 2690

# create n grams, starts from uni-gram, bi-gram, ..., n-gram
# [What]
# [What is]
# [What is love]
# [What is love by]
# [What is love by Twice]
sequences = []
for row in corpus:
    sequence = tokenizer.texts_to_sequences([row])[0]
    for i in range(1, len(sequence)):
        sequences.append(sequence[:i+1])
sequences = np.array(sequences)
print(sequences.shape)  # (12038,)
max_len = max([len(row) for row in sequences])
padded = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

xs, ys = padded[:, :-1], padded[:, -1]
ys = keras.utils.to_categorical(ys, num_classes=total_words)
print(np.array(xs).shape, np.array(ys).shape)  # (12038, 15) (12038, 2690)

print(tokenizer.word_index)  # {'the': 1, 'and': 2, 'i': 3, 'to': 4, 'a': 5, 'of': 6, 'my': 7, 'in': 8, 'me': 9 ...}
print(tokenizer.word_index['in'])  # 8
print(tokenizer.word_index['the'])  # 1
print(tokenizer.word_index['town'])  # 71
print(tokenizer.word_index['of'])  # 6
print(tokenizer.word_index['athy'])  # 713
print(tokenizer.word_index['one'])  # 39
print(tokenizer.word_index['jeremy'])  # 1790
print(tokenizer.word_index['lanigan'])  # 1791
print(xs[5], ys[5])  # [0 0 0 0 0 0 0 0 0 51 12 96 1217 48 2] [0. 0. 0. ... 0. 0. 0.]
print(xs[6], ys[6])  # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 2] [0. 0. 0. ... 0. 0. 0.]

model = keras.models.Sequential([
    keras.layers.Embedding(total_words, 64, input_length=max_len-1),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    # keras.layers.Conv1D(128, 5, activation='relu'),
    # keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(total_words, activation='softmax')
])
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])
model.fit(xs, ys, epochs=300)

reverse_word_index = dict([(index, word) for (word, index) in tokenizer.word_index.items()])
seed = "I've got a bad feeling about this"
for _ in range(200):
    seed_pad = tokenizer.texts_to_sequences([seed])
    seed_pad = keras.preprocessing.sequence.pad_sequences(seed_pad, maxlen=max_len-1)
    predict = model.predict(seed_pad)
    predict_word = reverse_word_index[np.argmax(predict)]
    seed += " "+predict_word
print(seed)
"""
Epoch 297/300
377/377 [==============================] - 3s 8ms/step - loss: 0.5648 - accuracy: 0.8477
Epoch 298/300
377/377 [==============================] - 4s 9ms/step - loss: 0.5609 - accuracy: 0.8446
Epoch 299/300
377/377 [==============================] - 4s 10ms/step - loss: 0.5529 - accuracy: 0.8456
Epoch 300/300
377/377 [==============================] - 3s 8ms/step - loss: 0.5510 - accuracy: 0.8468
I've got a bad feeling about this was died strolling moonlight are dim belfast is 
huff huff pure toome heartfrom easter polkas heartfrom crowds heartfrom there do see 
to save it sat on and it is tell to find the weirs drown of a tear for by the hand he 
bay die in down a kerry true oak part merry toome today to your provost and light i 
was the mountain side the moonlight crystal sighed workin by by proud saxon i our mountain 
wild are side by by night proud the from dawn dawn tory reminded tune nest chirping the 
good bubblin tie parlour the land of my word friends the gay weary ground reel door to 
a a jail collar reel back reel back craw craw was adoration roam by to me from them and the 
bold deceiver spancil hill i pain weary mans huff i sighed for tough fair much out from 
five our father jail row a good belfast victory sinking guard reel foaming nonsense finea 
finea color jail tie goggles huff finea o death glass leave all to my true armless jewel 
when is out and it makes my true native are ill be agin the wearin before spancil heartfrom 
sinking gown
"""