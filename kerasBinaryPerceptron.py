#!/usr/bin/env python
"""binary perceptron recreated with keras"""
from keras.models import Sequential
from keras.layers import Dense, Activation

from createData import *
import numpy as np


# variables
dims, samples = 2, 1000

# generate training data
data, test = createData(2, dims, samples)

points_train = np.array([d.position for d in data], np.float64)
labels_train = np.array([d.label for d in data], np.float64)

points_test = np.array([t.position for t in test], np.float64)
labels_test = np.array([t.label for t in test], np.float64)


# create noddy model
model = Sequential()
model.add(Dense(units=1, input_dim=dims, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# x_train and y_train are Numpy arrays
model.fit(points_train, labels_train, epochs=50, batch_size=20)


# print metrics
score = model.evaluate(points_test, labels_test, batch_size=128, verbose=1)
print 'accuracy =', score[1]
print 'loss =', score[0]
