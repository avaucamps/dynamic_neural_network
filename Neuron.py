import numpy as np 

class Neuron:
    def __init__(self, activation_function = None):
        if activation_function != None:
            self.activation_function = activation_function

    
    def run(self, inputs):
        return self.activation_function(inputs)