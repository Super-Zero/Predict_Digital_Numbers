# Prediction of Digital Numbers

A Multi-Layer Perceptron (MLP) implementation with PyTorch to predict handwritten numbers using the MNIST dataset.
The model is conformed with the following layers:
- The input layer has 784 input nodes
- The hidden layer 1 has 128 neurons and a ReLU activation function
- The hidden layer 2 has 64 neurons and a ReLU activation function
- The ouput layer has 10 neurons with an a SoftMax activation function

The file digits.py contains the MLP Network implementation with PyTorch.

The file helper.py contains visualization functions to visualise the results and the input images using Matplotlib


The image below shows a hadnwriten number on the left and on the right shows its class probability precited by the MLP network.

<img src="images/image-01.png">
