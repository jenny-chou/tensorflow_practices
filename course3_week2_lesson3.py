import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import io

imbd, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
train_data, test_data = imbd['train'], imbd['test']

tokenizer = info.features['text'].encoder
print(tokenizer.subwords)
"""
['the_', ', ', '. ', 'a_', 'and_', 'of_', 'to_', 's_', 'is_', 'br', 'in_', 'I_', 'that_', 'this_', 'it_', ' /><', ...
"""

sample_string = "Tensorflow, from basics to mastery"
tokenized_string = tokenizer.encode(sample_string)
print("Tokenized string is {}".format(tokenized_string))
original_string = tokenizer.decode(tokenized_string)
print("Original string is {}".format(original_string))
"""
Tokenized string is [6307, 2327, 2934, 2, 48, 4249, 4429, 7, 2652, 8050]
Original string is Tensorflow, from basics to mastery
"""

for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer.decode([ts])))
"""
6307 ----> Ten
2327 ----> sor
2934 ----> flow
2 ----> , 
48 ----> from 
4249 ----> basi
4429 ----> cs 
7 ----> to 
2652 ----> master
8050 ----> y
"""

BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_data.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))

print(tokenizer.vocab_size)  # 8185
embedding_dim = 64
model = keras.models.Sequential([
    keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 64)          523840    
_________________________________________________________________
global_average_pooling1d (Gl (None, 64)                0         
_________________________________________________________________
dense (Dense)                (None, 6)                 390       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 7         
=================================================================
Total params: 524,237
Trainable params: 524,237
Non-trainable params: 0
_________________________________________________________________
"""

num_epochs = 10
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
"""
Epoch 1/10
391/391 [==============================] - 10s 27ms/step - loss: 0.6730 - accuracy: 0.6264 - val_loss: 0.6218 - val_accuracy: 0.7614
Epoch 2/10
391/391 [==============================] - 10s 26ms/step - loss: 0.5173 - accuracy: 0.8119 - val_loss: 0.4438 - val_accuracy: 0.8442
Epoch 3/10
391/391 [==============================] - 10s 26ms/step - loss: 0.3687 - accuracy: 0.8743 - val_loss: 0.3606 - val_accuracy: 0.8686
Epoch 4/10
391/391 [==============================] - 10s 26ms/step - loss: 0.2964 - accuracy: 0.8956 - val_loss: 0.3334 - val_accuracy: 0.8660
Epoch 5/10
391/391 [==============================] - 10s 27ms/step - loss: 0.2591 - accuracy: 0.9072 - val_loss: 0.3196 - val_accuracy: 0.8722
Epoch 6/10
391/391 [==============================] - 11s 28ms/step - loss: 0.2311 - accuracy: 0.9186 - val_loss: 0.3160 - val_accuracy: 0.8762
Epoch 7/10
391/391 [==============================] - 10s 26ms/step - loss: 0.2127 - accuracy: 0.9237 - val_loss: 0.3140 - val_accuracy: 0.8734
Epoch 8/10
391/391 [==============================] - 10s 26ms/step - loss: 0.1960 - accuracy: 0.9296 - val_loss: 0.3090 - val_accuracy: 0.8809
Epoch 9/10
391/391 [==============================] - 10s 26ms/step - loss: 0.1818 - accuracy: 0.9353 - val_loss: 0.3162 - val_accuracy: 0.8797
Epoch 10/10
391/391 [==============================] - 10s 25ms/step - loss: 0.1690 - accuracy: 0.9420 - val_loss: 0.3194 - val_accuracy: 0.8795
"""

# e = model.layers[0]
# weights = e.get_weights()[0]
# print(weights.shape) # shape: (vocab_size, embedding_dim)
#
# out_v = io.open('course3_week2_lesson3_vecs.tsv', 'w', encoding='utf-8')
# out_m = io.open('course3_week2_lesson3_meta.tsv', 'w', encoding='utf-8')
# for word_num in range(1, tokenizer.vocab_size):
#   word = tokenizer.decode([word_num])
#   embeddings = weights[word_num]
#   out_m.write(word + "\n")
#   out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
# out_v.close()
# out_m.close()