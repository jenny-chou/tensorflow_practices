import json
from tensorflow import keras

num_vocab = 10000
with open("..\pythonProject\Sarcasm_Headlines_Dataset.json", 'r') as file:
    datastore = json.load(file)

sentences, labels, urls = [], [], []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)  # sentences is list of strings
word_index = tokenizer.word_index
print(len(word_index))  # word_index gives index number to ALL unique words in fitted text regardless of num_words
# 29657
sequences = tokenizer.texts_to_sequences(sentences)
# only the most frequent num_words will be converted, discard others
# sequence is list of lists (2D list), has shape (26709, None), each inner list has different length becasuse each string has different length
# sequence[0] has length 12, [1] has 14, [5] has 2
padded = keras.preprocessing.sequence.pad_sequences(sequences, padding='post', truncating='post')
print(padded[0], type(padded))
# [ 308    1  679 3337 2298   48  382 2576    1    6 2577 8434    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0] <class 'numpy.ndarray'>
print(padded.shape)  # (26709, 40)