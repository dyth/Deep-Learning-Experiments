#!/usr/bin/env python
"""binary perceptron recreated with keras"""
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

from createData import *
import numpy as np


# variables
labels, dims, samples = 3, 10, 10000


# generate training data
data, test = createData(labels, dims, samples)

points_train = np.array([d.position for d in data], np.float64)
labels_train = to_categorical([d.label for d in data], num_classes = labels)

points_test = np.array([t.position for t in test], np.float64)
labels_test = to_categorical([t.label for t in test], num_classes = labels)


# create noddy model
model = Sequential()
model.add(Dense(units=labels, input_dim=dims, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# x_train and y_train are Numpy arrays
model.fit(points_train, labels_train, epochs=50, batch_size=128)


# print metrics
score = model.evaluate(points_test, labels_test, batch_size=128, verbose=1)
print 'accuracy =', score[1]
print 'loss =', score[0]
