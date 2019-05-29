import numpy as np
from utils.utils import index_nan_argmax

class MaxPoolLayer:
    def __init__(self, window_size, stride=1):
        """
        Creates a MaxPool Layer.

        Arguments:
            - window_size: the size of the windpw (ex: (2,2)).
            - stride: int value.
        """
        self.window_size = window_size
        self.stride = stride
        self.shape = None

    
    def execute_forward_pass(self, input_array):
        """
        Returns the result of the maxpooling applied to the array passed in parameter.

        Arguments:
            - input_array: 3D array (input_index, input_width, input_height).

        Returns:
            - 3D array obtained after performing maxpool on input array.
        """
        self.input = input_array
        (input_width, input_height) = input_array.shape[1:]
        output_width = int((input_width - self.window_size[0]) / self.stride) + 1
        output_height = int((input_height - self.window_size[1]) / self.stride) + 1
        output = np.empty((input_array.shape[0], output_height, output_width))

        output_x_index = 0
        output_y_index = 0
        height_bound = input_height - self.window_size[1]
        width_bound = input_width - self.window_size[0]
        for input_index in range(input_array.shape[0]):
            for y in range(0, height_bound, self.stride):
                for x in range(0, width_bound, self.stride):
                    output[input_index, output_y_index, output_x_index] = np.max(
                        input_array[input_index, y:y+self.window_size[1], x:x+self.window_size[0]]
                    )
                    output_x_index += 1

                output_y_index += 1
                output_x_index = 0
            output_y_index = 0

        self.output = output
        return self.output

    
    def execute_backward_pass(self, d_pool):
        """
        Calculates gradient of the cost with respect to the output of this layer.

        Arguments:
            - d_pool: the gradient of the cost with respect to the output of the next layer.

        Returns:
            - the gradient of the cost with respect to the output of this layer.
        """
        doutput = np.zeros(self.input.shape)
        
        output_x = output_y = 0
        for input_index in range(self.input.shape[0]):
            for y in range(0, self.input.shape[1], self.stride):
                for x in range(0, self.input.shape[2], self.stride):
                    (a, b) = index_nan_argmax(self.input[input_index, y:y+self.window_size[1], x:x+self.window_size[0]])
                    doutput[input_index, y+a, x+b] = d_pool[input_index, output_y, output_x]
                    output_x += 1
                output_x = 0
                output_y += 1
            output_y = 0

        return doutput