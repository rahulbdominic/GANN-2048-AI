import numpy as np

# Main network class
class Network(object):
    def __init__(self, sizes, seed):

        ''' This takes the sizes as an input and generates random weights and
            and biases where the mean is 0 '''
        self.n_layers = len(sizes)
        self.sizes = sizes

        # Seed numpy random
        np.random.seed(seed)
        self.biases = [np.random.randn(1, y) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def getsizes(self):
        return self.sizes

    def getweights(self):
        return self.weights

    def getbiases(self):
        return self.biases

    def setweights(self, weights):
        self.weights = weights

    def setbiases(self, biases):
        self.biases = biases

    def feed_forward(self, activation):
        ''' Feed the initial inputs through the network iteratively and return
            the final outputs '''
        for b, w in zip(self.biases, self.weights):
            activation = sigmoid(np.dot(activation, w) + b)
        return activation

    def evaluate(self, current_board):
        ''' Feed the inputs through the feed_forward function and take the
            argmax of the outputs and return the action_matrix '''
        action_matrix = [np.argmax(self.feed_forward(current_board))]
        return action_matrix

# Sigmoid function
def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-val))

# Derivative of the sigmoid function
def sigmoid_differentiated(val):
    return sigmoid(z) * (1 - sigmoid(val))
