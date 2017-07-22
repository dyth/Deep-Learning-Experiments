# Deep Learning Experiments

The repository provides an exploratory framework from the evolutionary backbone of Deep Learning.

With regards to classification, a binary perceptron is trained, then a multiperceptron, multilayer perceptron, then convolutional neural network.

## Datapoint Generation
All data generated can be done for an arbitrary number of points in arbitrary dimensionality.
* `createData.py`: create random points, then group by nearest neighbours on randomly generated centroids.
* `createXOR.py`: create points in taxicab geometry where all the elements in the point are in {-1, 1}. The parity of the product of the all of the scalars determines which class the point belongs to.

## Keras
1. `perceptron.py`: perceptron in Keras trained on binary data created by `createData.py` with 2 labels. The dimensionality and number of sample points can be changed.
2. `multiPerceptron.py`: multiperceptron operating on MAP, classifying data created by `createData.py`. The dimensionality of the problem, number of labels and number of sample points can be changed.
3. `multilayerPerceptron.py`: multilayer perceptron which can solves the XOR problem for an arbitrary number of dimensions.
4. `mnistLeNet.py`: an implementation of LeNet-5 (LeCun et al, 1998), but with inputs as (28, 28) instead.
