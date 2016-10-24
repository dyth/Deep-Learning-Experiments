#!/usr/bin/env python

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os

# adapted from http://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/
# example usage python simple_neural_network -a 3072, 768, 384, 2 -d path/to/dataset

# resizes image to 32 by 32 pixels wide
def createFeatureVector(image, size=(32, 32)):
    return cv2.resize(image, size).flatten()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-a", "--architecture", required=True, help="no of neurons in layers")
args = vars(ap.parse_args())
 
# start data array, label list, get imagepaths
print("Describing images...")
imagePaths, data, labels = list(paths.list_images(args["dataset"])), [], []

# get image (path/class.image_num.jpg), construct feature vector, place in data
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    labels.append(imagePath.split(os.path.sep)[-1].split(".")[0])
    data.append(createFeatureVector(image))
    if i % 1000 == 0:
        print(" -- processed {}/{}".format(i, len(imagePaths)))
        
# preprocessing: normalise pixels to [0,1], change lables from strings to integers.
data = np.array(data) / 255.0
labels = np_utils.to_categorical(LabelEncoder().fit_transform(labels), 2)

# partition data into 75% training, 25% testing
print("Partitioning data ...")
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.25, random_state=42)
print("Done")

model = Sequential()
architecture = args["architecture"].split(",")
model.add(Dense(int(architecture[0]), input_dim=3072, init="uniform", activation="relu"))
for i in range(len(architecture)-2):
    model.add(Dense(int(architecture[i+1]), init="uniform", activation="relu"))
model.add(Dense(int(architecture[-1])))
model.add(Activation("softmax"))

# training using Stochastic Gradient Descent
print("Compiling model ...")
model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.01), metrics=["accuracy"])
model.fit(trainData, trainLabels, nb_epoch=50, batch_size=128, verbose=1)

# testing data
print("Evaluating testing set ...")
loss, accuracy = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
print("    loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
