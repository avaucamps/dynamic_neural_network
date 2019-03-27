import numpy as np 

class FlattenLayer:
    def execute_forward_pass(self, input_array):
        self.n_inputs, self.width, self.height = input_array.shape
        return input_array.reshape((self.n_inputs * self.width * self.height, 1))

    
    def execute_backward_pass(self, derror, weights):
        return weights.T.dot(derror).reshape([self.n_inputs, self.width, self.height])