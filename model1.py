"""
Model 1: Using raw programming without using any predefined libraries to create, train or test the models.
"""
from random import random
from math import exp
import numpy as np
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from random import random
import pandas as pd

# Fetching the MNIST dataset and splitting into training and testing sets
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=60000, test_size=10000, shuffle=False)

# Converting X data to float 32 and normalizing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255.0
X_test = X_test/255.0


def get_one_hot(targets):
    """ Helper function to return one-hot vectors for data labels. Uses pandas library.
    Parameters:
        targets: The training labels
    Returns:
        One-hot vectors of training labels
    """
    return pd.get_dummies(targets).values


# Converting desired outputs to 1-hot vectors
# Eg. "5" --> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
y_train = get_one_hot(y_train)
y_test = get_one_hot(y_test)
# Converting desired output to uint8
y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)


class MLP():
    def __init__(self, n_inputs=784, hidden_layers=[64], n_outputs=10):
        """Constructor for the multilayer perceptron class.
        Parameters:
            n_inputs (int): The amount of inputs
            hidden_layers (list): The amount of hidden nodes/layers (since it's only 1 hidden layer, theres only 1 element)
            n_outputs (int): The amount of outputs
        """
        self.n_inputs = n_inputs
        self.hidden_layers = hidden_layers
        self.n_outputs = n_outputs

        # Represent the layers
        layers = [n_inputs] + hidden_layers + [n_outputs]

        # Assign random weights, multiplying by 0.1 to keep them reasonably small
        weights = []
        for i in range(len(layers) - 1):
            weight = np.random.rand(layers[i], layers[i + 1]) * 0.1
            weights.append(weight)
        self.weights = weights

        # Saving the derivatives for each layer
        derivs = []
        for i in range(len(layers) - 1):
            deriv = np.zeros((layers[i], layers[i + 1]))
            derivs.append(deriv)
        self.derivatives = derivs

        # Saving the activations for each layer
        activations = []
        for i in range(len(layers)):
            activation = np.zeros(layers[i])
            activations.append(activation)
        self.activations = activations

    def forward_propagate(self, inputs):
        """Does forward propagation of the neural network from the inputs
        Parameters:
            inputs (ndarray): Inputs
        Returns:
            activations (ndarray): Outputs
        """

        # Set input layer activation to the input
        activations = inputs

        # Set the activations for backpropogation
        self.activations[0] = inputs

        # Iterating through layers
        for i, w in enumerate(self.weights):
            # determine matrix multiplication for previous activation and weight
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # set the activations for backpropogation
            self.activations[i + 1] = activations

        # returns the final output layer activation
        return activations

    def back_propagate(self, err):
        """Backpropogates error signal.
        Parameters:
            err (ndarray): The error
        Returns:
            err (ndarray): The final error of the input
        """

        # iterating backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = err * self._sigmoid_derivative(activations)

            # restructure delta to a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            curr_activations = self.activations[i]

            # restructure activations array to a 2d column matrix
            curr_activations = curr_activations.reshape(
                curr_activations.shape[0], -1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(curr_activations, delta_re)

            # backpropogate the next error
            err = np.dot(delta, self.weights[i].T)

    def train(self, X, Y, epochs, lr):
        """Trains model running forward prop and backprop
        Parameters:
            X (ndarray): Training datapoints
            Y (ndarray): Desired outputs
            epochs (int): Amount of epochs to train the network
            lr (float): Learning rate
        """
        # now enter the training loop
        for i in range(epochs):
            error_sum = 0

            # iterate through all the training data
            for j, input in enumerate(X):
                y = Y[j]

                # activate the network!
                output = self.forward_propagate(input)

                error = y - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(lr)

                # keep track of the MSE for reporting later
                error_sum += self._mse(y, output)

            # Epoch complete, report the training error
            print("Epoch {} - Training Loss: {}".format(i+1, error_sum/len(X)))

        print("Training complete!")
        print("=====")

    def gradient_descent(self, lr):
        """Performs gradient descend in the network
        Parameters:
            lr (float): The learning rate
        """
        # update weights by descending gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivs = self.derivatives[i]
            weights += derivs * lr

    def _sigmoid(self, x):
        """Sigmoid activation function
        Parameters:
            x (float): the input
        Returns:
            y (float): the output
        """

        y = 1.0 / (1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):
        """Sigmoid derivative function
        Parameters:
            x (float): the input
        Returns:
            y (float): the output
        """
        return x * (1.0 - x)

    def _mse(self, y, pred):
        """Mean Squared Error loss function
        Parameters:
            y (ndarray): ground truth
            pred (ndarray): predicted values
        Returns:
            (float): the output
        """
        return np.average((y - pred) ** 2)


if __name__ == "__main__":
    # Build MLP
    mlp = MLP(784, [64], 10)
    mlp.train(X_train, y_train, 10, 0.1)

    # Construct empty confusion matrix
    c_m = np.zeros(shape=(10, 10)).astype(int)

    # Test on testing data
    correct_guesses = 0
    for i in range(len(X_test)):
        pred = mlp.forward_propagate(X_test[i])
        pred = np.where(pred == np.amax(pred))[0][0]
        y = y_test[i]
        y = np.where(y == 1)[0][0]

        # print("Predicted:{} Expected:{}".format(pred, y))

        if pred == y:
            correct_guesses += 1
        c_m[pred][y] += 1

    print("Accuracy " + str(correct_guesses / 10000))
    print(c_m)
