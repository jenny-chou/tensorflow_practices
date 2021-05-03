import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

############## for some reason can't pull the complete dataset to cache for LSTM and GRU training #####################

imbd, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
print(imbd, info)
train_data, test_data = imbd['train'], imbd['test']

BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_data.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))

tokenizer = info.features['text'].encoder

embedding_dim = 64
model = keras.models.Sequential([
    keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    keras.layers.Conv1D(128, 5, activation=tf.nn.relu),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
# model.summary()
# num_epochs = 10
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 64)          523840    
_________________________________________________________________
conv1d (Conv1D)              (None, None, 128)         41088     
_________________________________________________________________
global_average_pooling1d (Gl (None, 128)               0         
_________________________________________________________________
dense (Dense)                (None, 6)                 774       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 7         
=================================================================
Total params: 565,709
Trainable params: 565,709
Non-trainable params: 0
_________________________________________________________________

Epoch 1/10
391/391 [==============================] - 79s 201ms/step - loss: 0.4966 - accuracy: 0.7480 - val_loss: 0.3317 - val_accuracy: 0.8704
Epoch 2/10
391/391 [==============================] - 91s 232ms/step - loss: 0.2491 - accuracy: 0.9094 - val_loss: 0.3230 - val_accuracy: 0.8739
Epoch 3/10
391/391 [==============================] - 38s 96ms/step - loss: 0.1892 - accuracy: 0.9324 - val_loss: 0.3214 - val_accuracy: 0.8763
Epoch 4/10
391/391 [==============================] - 21s 52ms/step - loss: 0.1549 - accuracy: 0.9460 - val_loss: 0.3902 - val_accuracy: 0.8691
Epoch 5/10
391/391 [==============================] - 19s 50ms/step - loss: 0.1292 - accuracy: 0.9547 - val_loss: 0.4092 - val_accuracy: 0.8694
Epoch 6/10
391/391 [==============================] - 19s 48ms/step - loss: 0.1102 - accuracy: 0.9631 - val_loss: 0.5132 - val_accuracy: 0.8548
Epoch 7/10
391/391 [==============================] - 19s 48ms/step - loss: 0.0911 - accuracy: 0.9699 - val_loss: 0.5537 - val_accuracy: 0.8562
Epoch 8/10
391/391 [==============================] - 19s 48ms/step - loss: 0.0779 - accuracy: 0.9746 - val_loss: 0.6060 - val_accuracy: 0.8558
Epoch 9/10
391/391 [==============================] - 18s 46ms/step - loss: 0.0718 - accuracy: 0.9774 - val_loss: 0.6241 - val_accuracy: 0.8553
Epoch 10/10
391/391 [==============================] - 18s 47ms/step - loss: 0.0553 - accuracy: 0.9830 - val_loss: 0.6843 - val_accuracy: 0.8549
"""


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
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, None, 64)          523840    
_________________________________________________________________
bidirectional (Bidirectional (None, 128)               66048     
_________________________________________________________________
dense_2 (Dense)              (None, 6)                 774       
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 7         
=================================================================
Total params: 590,669
Trainable params: 590,669
Non-trainable params: 0
_________________________________________________________________


"""


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
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, None, 64)          523840    
_________________________________________________________________
bidirectional (Bidirectional (None, None, 128)         66048     
_________________________________________________________________
bidirectional_1 (Bidirection (None, 64)                41216     
_________________________________________________________________
dense_2 (Dense)              (None, 6)                 390       
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 7         
=================================================================
Total params: 631,501
Trainable params: 631,501
Non-trainable params: 0
_________________________________________________________________


"""
