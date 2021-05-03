import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

"""
# NLP practice with simple sentences
sentences = [
    "I love my dog",
    "I love my cat",
    "You love my dog!",
    "Do you think my dog is amazing?"
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

test_sentences = [
    "I really love my dog",
    "My dog loves my manatee"
]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)
print(test_padded)
"""

# # generate new text based on this Irish poetry
# multi-class classification problem: which word in the words pool is most likely to be the next word of this sentence?
with open("Irish_lyrics.txt", 'r') as file:
    lyrics = file.read()
lyrics = lyrics.lower().split('\n')

# create Tokenizer object and fit with the Irish poetry
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(lyrics)

# total_words contains the number of unique words/tokens plus 1 for the zero padding, 0, for later use
total_words = len(tokenizer.word_index)+1

# generate multiple sequences from each sentence
input_seq = []
for sentence in lyrics:
    seq = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(seq)):
        n_gram_seq = seq[:i + 1]
        input_seq.append(n_gram_seq)

# max_seq_len contains the length of longest sentence, and pad all sequences to that length
max_seq_len = max([len(seq) for seq in input_seq])
input_seq = pad_sequences(input_seq, maxlen=max_seq_len, padding='pre')

# output is the last word of the sequence, rest of the sequence is the input
xs = input_seq[:,:-1]
labels = input_seq[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# create bidirectional LSTM model and compile and fit
model3 = Sequential([
    Embedding(total_words, 240, input_length=max_seq_len-1),  # because we used the last column as y
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    Dense(total_words, activation='softmax')  # because we want to know which word comes out as result from our words pool
])
model3.summary()
"""
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, 15, 240)           645840    
_________________________________________________________________
bidirectional_2 (Bidirection (None, 300)               469200    
_________________________________________________________________
dense_4 (Dense)              (None, 2691)              809991    
=================================================================
Total params: 1,925,031
Trainable params: 1,925,031
Non-trainable params: 0
"""
adam = tf.keras.optimizers.Adam(lr=0.01)
model3.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model3.fit(xs, ys, epochs=30, batch_size=batch_size)

# create new seed and generate new text
seed = "I made a poetry"
pred_words_len = 200
pred_words = "I made a poetry"

for _ in range(pred_words_len):
    token_list = tokenizer.texts_to_sequences([seed])
    token_list = pad_sequences(token_list, maxlen=max_seq_len-1, padding='pre')
    prediction = model3.predict(token_list)
    prediction = np.argmax(prediction)
    for word, index in tokenizer.word_index.items():
        if index == prediction:  # notice 0 is for padding and 1 is the first word. So index of all classes range from 1 to total_words
            pred_words += " " + word
            seed += " " + word
    seed = seed[1:]
print(pred_words)
"""
I made a poetry board a plain country girl i hear someone tapping tapping rings 
aisey the golden variety rings find upon where but law while golden alone thru 
rings shadow the girls alone glance turns brown hair axe mcbryde bragh road to 
stand behind him love them golden coins did stray from buttoned and awake years 
gone away ye grows on your eyes twinkle bright heavens cruel feet goodbye my 
darling while mine inheritance now gone rigadoo ringing and love grows on wild 
shed love love gone credit she is when colonel wid you gone away your love 
letters and light an guard heart will gone weeping and heavens had feet begin 
to enjoy gone flood the floor that when she had gone temper i elf country oer 
the bold deceiver love weeping shoulder awhile hour old times house gone by the
foots stirring evening moves over thine own ever gone away ye words ago stand 
love weeping seen than any one whos me gone and heavens played welcome friend 
than to enjoy you stole before mcbryde ringing and now gone by and how weeping 
easily have gone and rising preacher and while gone oh then up at brooks 
academy gone by when
"""