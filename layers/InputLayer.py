import numpy as np

class InputLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape


    def execute_forward_pass(self, input_vector):
        return input_vector

    
    def get_output_shape(self):
        return self.input_shape