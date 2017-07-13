#!/usr/bin/env python
"""randomly create data of dimensionality and labels"""
import random, math

# an easy to access structure
class point:
    def __init__(self, position, label):
        'store position and coordinates'
        self.position = position
        self.label = label



def randomList(n):
    'return list of len(n) of randomly generated floats in range {0, 1}'
    return [random.random() for _ in range(n)]


def euclideanDistance(position, centroid):
    'return euclidean distance between two points'
    distance = 0.0
    for i in range(len(position)):
        distance += (position[i] - centroid[i]) ** 2
    return math.sqrt(distance)


def classify(position, centroids):
    'return nearest centroid based on Euclidean Distance'
    distances = [euclideanDistance(position, centre) for centre in centroids]
    return distances.index(min(distances))


def createData(labels, dimensionality, samples):
    'create len(samples) of dimensionality in grouped into labels no of classes'
    centroids = [randomList(dimensionality) for _ in range(labels)]
    # create data
    data = [randomList(dimensionality) for _ in range(samples)]
    data = [point(datum, classify(datum, centroids)) for datum in data]
    # create testing data
    test = min(100, samples / 10)
    test = [randomList(dimensionality) for _ in range(test)]
    test = [point(datum, classify(datum, centroids)) for datum in test]
    return data, test


if __name__ == "__main__":
    labels = 2
    dimensionality = 2
    samples = 10

    for point in createData(labels, dimensionality, samples):
        print point.position, point.label
