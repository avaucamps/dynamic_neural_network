import numpy as np
from Neuron import Neuron
from utils.utils import truncated_normal

class HiddenLayer:
    def __init__(self, n_neurons, input_shape, activation_function, learning_rate, is_agent_mode_enabled = False):
        self.activation_function = activation_function
        self.is_agent_mode_enabled = is_agent_mode_enabled
        self.output = np.empty([n_neurons, 1])

        if self.is_agent_mode_enabled:
            self.setup_neurons(n_neurons, input_shape, activation_function)
            self.weights = np.empty([n_neurons, input_shape])
        else:
            self.setup_weights(n_neurons, input_shape)
        
        self.learning_rate = learning_rate


    def setup_neurons(self, n_neurons, input_shape, activation_function):
        self.neurons = []
        for _ in range(n_neurons):
            self.neurons.append(Neuron(input_shape, activation_function))


    def get_n_parameters(self):
        if self.is_agent_mode_enabled:
            return self.neurons[0].get_n_parameters() * len(self.neurons)

        return self.weights.shape[0] * self.weights.shape[1]


    def get_output_shape(self):
        return self.output.shape[0]

    
    def setup_weights(self, n_neurons, input_shape):
        rad = 1 / np.sqrt(input_shape)
        X = truncated_normal(mean=0, 
                             sd=1, 
                             low=-rad, 
                             upp=rad)
        self.weights = np.array(X.rvs((n_neurons, input_shape)))

    
    def execute_forward_pass(self, inputs):
        self.inputs = inputs

        if self.is_agent_mode_enabled:
            for i in range(len(self.neurons)):
                self.output[i] = self.neurons[i].run(inputs)
        else:  
            self.output = self.activation_function(np.dot(self.weights, inputs))

        return self.output


    def execute_backward_pass(self, error, next_layer_weights):
        current_layer_error = np.dot(next_layer_weights.T, error)
        gradient = current_layer_error * self.activation_function(self.output, derivative=True)
        gradient = np.dot(gradient, self.inputs.T)

        if self.is_agent_mode_enabled:
            for i in range(len(self.neurons)):
                self.weights[i] = self.neurons[i].update(self.learning_rate * gradient[i])    
        else:
            self.weights -= self.learning_rate * gradient
        
        return self.weights, current_layer_error

    
    def get_output(self):
        return self.output