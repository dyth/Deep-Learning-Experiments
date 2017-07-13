#!/usr/bin/env python
"""multiclass perceptron recreated with keras"""
from keras.models import Sequential
from keras.layers import Dense, Activation

from createData import *
import numpy as np


# generate training data
labels, dimensionality, samples = 2, 2, 1000
data, test = createData(labels, dimensionality, samples)

points_train = np.array([d.position for d in data], np.float64)
labels_train = np.array([d.label for d in data], np.float64)

points_test = np.array([t.position for t in test], np.float64)
labels_test = np.array([t.label for t in test], np.float64)


# create noddy model
model = Sequential()
model.add(Dense(units=2, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# x_train and y_train are Numpy arrays
model.fit(points_train, labels_train, epochs=50, batch_size=20)

score = model.evaluate(points_test, labels_test, batch_size=128, verbose=1)
print 'accuracy = ', score[1]
print 'loss = ', score[0]
