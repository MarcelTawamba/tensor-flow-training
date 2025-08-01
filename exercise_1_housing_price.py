"""
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%201%20-%20House%20Prices/Exercise_1_House_Prices_Question.ipynb
build a neural network that predicts the price of a house according to a simple formula: a house costs 50k + 50k per bedroom, so that a 1 bedroon house costs 100k, 2 bedroom costs 150k etc.

create a neural network that would learn this relationship so that it could predict the price of a house with 7 bedrooms.
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([1, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
model.fit(xs, ys, epochs=500)
print(model.predict(np.array([7.0], dtype=float)))
# the model should predict a value close to 4.0