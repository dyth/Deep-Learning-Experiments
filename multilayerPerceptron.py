#!/usr/bin/env python
"""multilayer perceptron trained on XOR problem"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

from createXOR import *
import numpy as np


# variables
dims, samples = 2, 2000


# generate training and testing data and their labels
(train_data, train_labels), (test_data, test_labels) = createData(dims, samples)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# create model
np.random.seed(1729)
model = Sequential()
model.add(Dense(units=2, input_dim=dims, activation='relu'))
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=2, activation='sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# train model
model.fit(train_data, train_labels, epochs=10, batch_size=10)


# print metrics
score = model.evaluate(test_data, test_labels, batch_size=len(test_labels), verbose=1)
print 'accuracy =', score[1]
print 'loss =', score[0]
