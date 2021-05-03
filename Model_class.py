import tensorflow as tf
import numpy as np
from tensorflow import keras

input = keras.Input(shape=(1,))
hidden = keras.layers.Dense(1, activation='relu')(input)
output = keras.layers.Dense(1, activation='sigmoid')(hidden)
model = keras.Model(inputs=input, outputs=output)
model.summary()
"""
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 1)]               0         
_________________________________________________________________
dense (Dense)                (None, 1)                 2         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 2         
=================================================================
Total params: 4
Trainable params: 4
Non-trainable params: 0
_________________________________________________________________
"""

model = keras.models.Sequential([
    keras.layers.Dense(1, input_shape=[1], activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_2 (Dense)              (None, 1)                 2         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 2         
=================================================================
Total params: 4
Trainable params: 4
Non-trainable params: 0
_________________________________________________________________
"""