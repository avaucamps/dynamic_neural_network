import numpy as np
from utils.utils import truncated_normal

class OutputLayer:
    def __init__(self, n_neurons, input_shape, activation_function, learning_rate):
        self.activation_function = activation_function
        self.setup_weights(n_neurons, input_shape)
        self.learning_rate = learning_rate

    
    def setup_weights(self, n_neurons, input_shape):
        rad = 1 / np.sqrt(input_shape)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights = np.array(X.rvs((n_neurons, input_shape)))

    
    def get_n_parameters(self):
        return self.weights.shape[0] * self.weights.shape[1]

    
    def execute_forward_pass(self, inputs):
        self.output = self.activation_function(np.dot(self.weights, inputs))
        return self.output


    def execute_backward_pass(self, error, previous_output):
        gradient = error * self.activation_function(self.output, derivative=True)
        gradient = np.dot(gradient, previous_output.T)
        self.weights -= self.learning_rate * gradient

        return self.weights