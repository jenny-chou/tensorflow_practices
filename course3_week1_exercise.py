import csv
from tensorflow import keras

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
with open("bbc-text.csv", 'r') as file:
    bbc = csv.reader(file, delimiter=',')
    next(bbc)  # skip header
    for row in bbc:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            sentence = sentence.replace(" "+word+" ", " ")
            sentence = sentence.replace("  ", " ")
        sentences.append(sentence)
print(len(sentences))  # 2225
print(sentences[0])
# tv future hands viewers home theatre systems plasma high-definition tvs digital video recorders moving
# living room # way people watch tv will radically different five years time. according expert panel gathered
# annual consumer # electronics show las vegas discuss new technologies will impact one favourite pastimes.

tokenizer = keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(len(word_index))  # 29714

sequences = tokenizer.texts_to_sequences(sentences)
padded = keras.preprocessing.sequence.pad_sequences(sequences, padding='post', truncating='post')
print(padded[0])  # [  96  176 1158 ...    0    0    0]
print(padded.shape)  # (2225, 2442)

label_tokenizer = keras.preprocessing.text.Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index
label_seq = label_tokenizer.texts_to_sequences(labels)  # .texts_to_sequences() output list of lists
print(label_seq)
print(label_word_index)
"""
[[4], [2], [1], [1], [5], [3], [3], [1], [1], [5], [5], [2], [2], [3], [1], [2], [3], [1], [2], ...

{'sport': 1, 'business': 2, 'politics': 3, 'tech': 4, 'entertainment': 5}
"""