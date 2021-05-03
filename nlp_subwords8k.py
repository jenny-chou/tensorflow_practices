import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import matplotlib.pyplot as plt


imbd, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
train_data, test_data = imbd['train'], imbd['test']

tokenizer = info.features['text'].encoder
print(tokenizer.subwords)

sample_string = "Tensorflow, from basics to mastery"
tokenized_string = tokenizer.encode(sample_string)
print("Tokenized string is {}".format(tokenized_string))
original_string = tokenizer.decode(tokenized_string)
print("Original string is {}".format(original_string))

for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer.decode([ts])))

BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_data.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))

print(tokenizer.vocab_size)
embedding_dim = 64
model = keras.models.Sequential([
    keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
num_epochs = 10
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
"""

"""
plt.plot(range(num_epochs), history.history['accuracy'], 'r', label="Training accuracy")
plt.plot(range(num_epochs), history.history['val_accuracy'], 'b', label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()

plt.plot(range(num_epochs), history.history['loss'], 'r', label="Training loss")
plt.plot(range(num_epochs), history.history['val_loss'], 'b', label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.figure()


model = keras.models.Sequential([
    keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
num_epochs = 10
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
"""

"""
plt.plot(range(num_epochs), history.history['accuracy'], 'r', label="Training accuracy")
plt.plot(range(num_epochs), history.history['val_accuracy'], 'b', label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()

plt.plot(range(num_epochs), history.history['loss'], 'r', label="Training loss")
plt.plot(range(num_epochs), history.history['val_loss'], 'b', label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.figure()


model = keras.models.Sequential([
    keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    keras.layers.Conv1D(128, 5, activation=tf.nn.relu),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
num_epochs = 10
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
"""

"""
plt.plot(range(num_epochs), history.history['accuracy'], 'r', label="Training accuracy")
plt.plot(range(num_epochs), history.history['val_accuracy'], 'b', label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()

plt.plot(range(num_epochs), history.history['loss'], 'r', label="Training loss")
plt.plot(range(num_epochs), history.history['val_loss'], 'b', label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.figure()
