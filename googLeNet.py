#!/usr/bin/env python
"""googLeNet v1, with batch instead of local response normalisation"""
from keras.models import Model
from keras.layers import Input, Dense, Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Dropout, ZeroPadding2D, Activation
from keras.layers import concatenate, Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.utils import plot_model

import numpy as np


def inception(input, l1, l3_1, l3_2, l5_1, l5_2, lP):
    'inception module, with l signifying the number of neurones in each tower'
    tower1 = Conv2D(l1, (1, 1), activation='relu')(input)

    tower3 = Conv2D(l3_1, (1, 1), activation='relu', padding='same')(input)
    tower3 = Conv2D(l3_2, (3, 3), activation='relu', padding='same')(tower3)

    tower5 = Conv2D(l5_1, (1, 1), activation='relu', padding='same')(input)
    tower5 = Conv2D(l5_2, (5, 5), activation='relu', padding='same')(tower5)

    towerP = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
    towerP = Conv2D(lP, (1, 1), activation='relu', padding='same')(towerP)

    return concatenate([tower1, tower3, tower5, towerP], axis=3)

"""
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
"""

# create model
inputs = Input(shape=(224, 224, 3))

# pre-inception layers
conv_7_2 = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
conv_7_2 = Activation('relu')(conv_7_2)

maxPool_3_2 = MaxPooling2D((3, 3), (2, 2), padding='same')(conv_7_2)

LRN = BatchNormalization()(maxPool_3_2)

conv_1_1 = Conv2D(192, (1, 1), strides=(1, 1), padding='same')(LRN)
conv_1_1 = Activation('relu')(conv_1_1)

conv_3_1 = Conv2D(192, (3, 3), strides=(1, 1), padding='same')(conv_1_1)
conv_3_1 = Activation('relu')(conv_3_1)

LRN = BatchNormalization()(conv_3_1)

maxPool_3_2 = MaxPooling2D((3, 3), (2, 2), padding='same')(LRN)


# inception layers
inception3a = inception(maxPool_3_2, 64, 96, 128, 16, 32, 32)
inception3b = inception(inception3a, 128, 128, 192, 32, 96, 64)

maxPool_3_2 = MaxPooling2D((3, 3), (2, 2), padding='same')(inception3b)

inception4a = inception(maxPool_3_2, 192, 96, 208, 16, 48, 64)
inception4b = inception(inception4a, 160, 112, 224, 32, 64, 64)
inception4c = inception(inception4b, 128, 128, 256, 32, 64, 64)
inception4d = inception(inception4c, 112, 144, 288, 32, 64, 64)
inception4e = inception(inception4d, 256, 160, 320, 32, 128, 128)

maxPool_3_2 = MaxPooling2D((3, 3), (2, 2), padding='same')(inception4e)

inception5a = inception(maxPool_3_2, 256, 160, 320, 32, 128, 128)
inception5b = inception(inception5a, 384, 192, 384, 48, 128, 128)


# auxillary classifiers
avgPool_5_1 = AveragePooling2D((5, 5), strides=(1, 1))(inception4a)
conv_3_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(avgPool_5_1)
flatten = Flatten()(conv_3_1)
dense = Dense(1024, activation='relu')(flatten)
dropout = Dropout(0.7)(dense)
class1 = Dense(1000, activation='softmax')(dropout)

avgPool_5_1 = AveragePooling2D((5, 5), strides=(1, 1))(inception4d)
conv_3_1 = Conv2D(128, (1, 1), padding='same', activation='relu')(avgPool_5_1)
flatten = Flatten()(conv_3_1)
dense = Dense(1024, activation='relu')(flatten)
dropout = Dropout(0.7)(dense)
class2 = Dense(1000, activation='softmax')(dropout)
                   

# final classifier
avgPool_7_1 = AveragePooling2D((7, 7), strides=(1, 1))(inception5b)
flatten = Flatten()(avgPool_7_1)
dropout = Dropout(0.4)(flatten)
outputs = Dense(1000, activation='softmax')(dropout)


# create model, summarize, compile and train
model = Model(inputs=inputs, outputs=[class1, class2, outputs])
plot_model(model, to_file='googLeNet.png', show_shapes=True)
model.summary()
"""
model.compile(optimizer='adam', loss='categorical_cross_entropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=10)


# evaluate
score = model.evaluate(x_test, y_test, verbose=0)
print 'accuracy =', score[1]
print 'loss =', score[0]
"""
