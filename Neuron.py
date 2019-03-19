import numpy as np 
from utils import truncated_normal

class Neuron:
    def __init__(self, input_shape, activation_function = None):
        self.setup_weights(input_shape)
        if activation_function != None:
            self.activation_function = activation_function


    def setup_weights(self, input_shape):
        rad = 1 / np.sqrt(input_shape)
        X = truncated_normal(mean=0, 
                             sd=1, 
                             low=-rad, 
                             upp=rad)
        self.weights = np.array(X.rvs((1, input_shape)))

    
    def get_n_parameters(self):
        return self.weights.shape[1]


    def get_weights(self):
        return self.weights

    
    def run(self, inputs):
        return self.activation_function(np.dot(self.weights, inputs))

    
    def update(self, gradient):
        self.weights -= gradient

        return self.weights