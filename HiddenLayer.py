import numpy as np
from Neuron import Neuron
from utils import truncated_normal

class HiddenLayer:
    def __init__(self, n_neurons, input_shape, activation_function, learning_rate):
        self.activation_function = activation_function
        self.setup_neurons(n_neurons, activation_function)
        self.setup_weights(n_neurons, input_shape)
        self.learning_rate = learning_rate


    def setup_neurons(self, n_neurons, activation_function):
        self.neurons = np.empty([n_neurons, 1], dtype=object)
        for i in range(n_neurons):
            self.neurons[i] = Neuron(activation_function)


    def get_output_shape(self):
        return len(self.neurons)

    
    def setup_weights(self, n_neurons, input_shape):
        rad = 1 / np.sqrt(input_shape)
        X = truncated_normal(mean=0, 
                             sd=1, 
                             low=-rad, 
                             upp=rad)
        self.weights = np.array(X.rvs((n_neurons, input_shape)))

    
    def execute_forward_pass(self, inputs):
        self.output = self.activation_function(np.dot(self.weights, inputs))
        self.inputs = inputs
        return self.output


    def execute_backward_pass(self, error, next_layer_weights):
        current_layer_error = np.dot(next_layer_weights.T, error)
        gradient = current_layer_error * self.activation_function(self.output, derivative=True)
        gradient = np.dot(gradient, self.inputs.T)
        self.weights -= self.learning_rate * gradient
        
        return self.weights, current_layer_error

    
    def get_output(self):
        return self.output