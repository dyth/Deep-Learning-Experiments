#!/usr/bin/env python
"""CNN for MNIST"""
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout, Activation
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.optimizers import SGD

import numpy as np


# input image dimensions
r, c = 28, 28
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# LeNet's input is (32, 32), but as it has been rescaled, train on (28, 28)
#x_train = np.pad(x_train, (2,2), 'constant')
#x_test = np.pad(x_test, (2,2), 'constant')

# reshape depending on backend
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, r, c).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, r, c).astype('float32')
    input_shape = (1, r, c)
else:
    x_train = x_train.reshape(x_train.shape[0], r, c, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], r, c, 1).astype('float32')
    input_shape = (r, c, 1)
x_train /= 255
x_test /= 255


# ensure equal dimensions of points and labels
length = min(len(x_train), len(y_train))
x_train, y_train = x_train[:length], y_train[:length]
length = min(len(x_test), len(y_test))
x_test, y_test = x_test[:length], y_test[:length]

# create matrix of one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# create model
model = Sequential()
# first conv layer
model.add(Conv2D(6, kernel_size=(5, 5), input_shape=(r, c, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# second conv layer
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# feedforward layer
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='sigmoid'))


# summarize, compile and train
#model.summary()
sgd = SGD(lr=0.02)
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=128)


# evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print 'accuracy =', score[1]
print 'loss =', score[0]
