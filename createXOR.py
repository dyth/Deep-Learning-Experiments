#!/usr/bin/env python
"""create binary classifiable data in the shape of an XOR gate"""
import random, operator
import numpy as np



def randomList(n):
    'return list of len(n) of randomly generated floats in range {-1, 1}'
    #return [random.uniform(-1.0, 1.0) for _ in range(n)]
    return [random.choice([-1.0, 1.0]) for _ in range(n)]

    
def classify(point):
    'return class based on sign of product of multiples'
    #return np.sign(reduce(operator.mul, point))
    return 0 if reduce(operator.mul, point) > 0 else 1


def createData(dims, samples):
    'create len(samples) of dimensionality in grouped into labels no of classes'
    # create and classify training data
    train_data = [randomList(dims) for _ in range(samples)]
    train_labels = [classify(t) for t in train_data]
    test_data = [randomList(dims) for _ in range(max(100, samples / 10))]
    test_labels = [classify(t) for t in test_data]
    
    # convert to numpy array
    train_data = np.array(train_data, np.float64)
    train_labels = np.array(train_labels, np.float64)
    test_data = np.array(test_data, np.float64)
    test_labels = np.array(test_labels, np.float64)

    return (train_data, train_labels), (test_data, test_labels)


if __name__ == "__main__":
    (X, T), _ = createData(2, 1000)

    import numpy as np
    import matplotlib.pyplot as plt

    x = [x[0] for x in X]
    y = [n[1] for n in X]
    
    color = ['r' if t == 1 else 'b' for t in T]
    
    plt.scatter(x, y, color=color)
    plt.show()
