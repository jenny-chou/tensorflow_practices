import tensorflow as tf
from tensorflow import keras
import numpy as np
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


data="In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a " \
     "man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho " \
     "didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and " \
     "the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I " \
     "might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a " \
     "cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon " \
     "arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for " \
     "the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting " \
     "the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in " \
     "Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans " \
     "Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia " \
     "and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how " \
     "the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks " \
     "at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree " \
     "long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for " \
     "Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe " \
     "stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the " \
     "girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young " \
     "Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried " \
     "Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil " \
     "he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at " \
     "the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too " \
     "much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen " \
     "stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at " \
     "Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim " \
     "McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the " \
     "piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, " \
     "in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."

corpus = data.lower().split('\n')  # corpus is a list of split strings/sentences
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # word_index starts with 1, +1 so the len() matches the max index number
print(tokenizer.word_index)
# {'and': 1, 'the': 2, 'a': 3, 'in': 4, 'all': 5, 'i': 6, 'for': 7, 'of': 8, 'lanigans': 9, ...}
print(total_words)
# 263

input_sequences = []
for row in corpus:
    sequence = tokenizer.texts_to_sequences([row])[0]
    for i in range(1, len(sequence)):
        input_sequences.append(sequence[:i+1])
print(np.array(input_sequences).shape)
# (453,) , this is to say each string has various length, equivalent to (453, None)
max_len = max([len(row) for row in input_sequences])
padded = np.array(keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_len))
print(padded.shape)
# (453, 11)

xs, labels = padded[:, :-1], padded[:, -1]
ys = keras.utils.to_categorical(labels, num_classes=total_words)

print(tokenizer.word_index)  # {'and': 1, 'the': 2, 'a': 3, 'in': 4, 'all': 5, 'i': 6,
print(tokenizer.word_index['in'])  # 4
print(tokenizer.word_index['the'])  # 2
print(tokenizer.word_index['town'])  # 66
print(tokenizer.word_index['of'])  # 8
print(tokenizer.word_index['athy'])  # 67
print(tokenizer.word_index['one'])  # 68
print(tokenizer.word_index['jeremy'])  # 69
print(tokenizer.word_index['lanigan'])  # 70
print(xs[5], ys[5])
"""
[ 0  0  0  0  4  2 66  8 67 68] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
"""
print(xs[6], ys[6])
"""
[ 0  0  0  4  2 66  8 67 68 69] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
"""

model = keras.models.Sequential([
    keras.layers.Embedding(total_words, 64, input_length=max_len-1),
    keras.layers.Bidirectional(keras.layers.LSTM(20)),
    keras.layers.Dense(total_words, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(xs, ys, epochs=500)

reverse_word_index = dict([(index, word) for (word, index) in tokenizer.word_index.items()])
seed = "Laurence went to dublin"
for _ in range(20):
    seed_pad = tokenizer.texts_to_sequences([seed])[0]
    # texts_to_sequences() takes in list of string(s). Thus it expects ["seed string ..."]
    # texts_to_sequences() returns list of list(s). Thus it returns [[1, 2, 3, 4, ...]]
    # texts_to_sequences()[0] returns [1, 2, 3, 4, ...]
    seed_pad = keras.preprocessing.sequence.pad_sequences([seed_pad], maxlen=max_len-1)
    # pad_sequences() expects list of list(s). Thus it expects [[1, 2, 3, 4, ...]]
    predict = model.predict(seed_pad)
    # predict has shape (number of class,) where number of classes is total_words
    # each entrance in prediction is a probability of this word being the next following word in this string
    predict_word = reverse_word_index[np.argmax(predict)]
    # find the index which has the highest probability, and use the lookup table reverse_word_index
    # to find the corresponding word, and that is our next word
    seed += " "+predict_word

print(seed)
"""
Epoch 499/500
15/15 [==============================] - 0s 6ms/step - loss: 0.1258 - accuracy: 0.9514
Epoch 500/500
15/15 [==============================] - 0s 5ms/step - loss: 0.1259 - accuracy: 0.9492
Laurence went to dublin his right with peggy mcgilligan mcgilligan glisten glisten glisten 
glisten glisten gray together at lanigans ball ball ball ball ball
"""