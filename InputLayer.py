import numpy as np 
from Neuron import Neuron

class InputLayer:
    def __init__(self, input_shape):
        self.setup_neurons(input_shape)


    def setup_neurons(self, input_shape):
        self.neurons = np.empty([input_shape, 1], dtype=object)
        for i in range(input_shape):
            self.neurons[i] = Neuron()

    
    def execute_forward_pass(self, input_vector):
        return input_vector