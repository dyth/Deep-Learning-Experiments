# Deep Learning Experiments

The repository provides an exploratory framework from the evolutionary backbone of Deep Learning.

With regards to classification, a binary perceptron is trained, then a multiperceptron, multilayer perceptron, then convolutional neural network.

## Keras
1. `kerasBinaryPerceptrion.py`: perceptron in Keras trained on binary data created by `createData.py`. The dimensionality and number of sample points can be changed.
2. `kerasMultiPerceptron.py`: multiperceptron operating on MAP, classifying data created by `createData.py`. The dimensionality of the problem, number of labels and number of sample points can be changed.
3. `feedforwardnetwork.py`: multilayer perceptron with variable architecture and dataset, run by `python feedforwardnetwork.py -a 3072,768,384,2 -d path/to/dataset`
