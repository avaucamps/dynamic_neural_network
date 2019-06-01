import numpy as np
from utils.utils import truncated_normal

class OutputLayer:
    def __init__(self, n_neurons, input_shape, activation_function, learning_rate):
        self.activation_function = activation_function
        self.input_shape = input_shape
        self.setup_weights(n_neurons, input_shape)
        self.learning_rate = learning_rate

    
    def setup_weights(self, n_neurons, input_shape):
        rad = 1 / np.sqrt(input_shape)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights = np.array(X.rvs((n_neurons, input_shape)))

    
    def get_init_weights(self, n_neurons, input_shape):
        rad = 1 / np.sqrt(input_shape)
        X = truncated_normal(mean=0, 
                             sd=1, 
                             low=-rad, 
                             upp=rad)
        return np.array(X.rvs((n_neurons, input_shape)))

    
    def get_output_shape(self):
        return self.weights.shape[0]


    def add_input(self):
        new_input = self.get_init_weights(self.get_output_shape(), 1)
        self.weights = np.concatenate((self.weights, new_input), axis=1)
        self.input_shape = self.weights.shape[1]

    
    def remove_inputs(self, index):
        if index > self.weights.shape[1]: return

        if index > 0 and index < self.weights.shape[1]-1:
            new_weights_1 = self.weights[:, :index]
            new_weights_2 = self.weights[:, index+1:]
            self.weights = np.concatenate((new_weights_1, new_weights_2), axis=1)
        elif index == 0:
            if self.weights.shape[1] == 1:
                self.weights = None
            else:
                self.weights = self.weights[:, 1:]
        elif index == self.weights.shape[1]-1:
            self.weights = self.weights[:, :index]


    def update_input_shape(self, input_shape):
        if self.input_shape == input_shape: return
        
        new_weights = np.empty([self.get_output_shape(), input_shape])
        if input_shape < self.input_shape:
            new_weights = self.weights[:, :input_shape]
        else:
            new_weights[:, :self.input_shape] = self.weights
            new_weights[:, self.input_shape:] = self.get_init_weights(self.get_output_shape(), input_shape - self.input_shape)

        self.input_shape = input_shape
        self.weights = new_weights

    
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