import numpy as np 
from utils import relu, relu_derivative

class ConvolutionLayer:
    def __init__(self, n_filters, filter_size, learning_rate, stride=1):
        """
        Creates a Convolutional Layer.

        Arguments:
            - n_filters: the number of filters the layer should have.
            - filter_size: the size of each filter (width, height).
            - stride: int value.
        """
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.filters = self.get_initialized_filters(n_filters, filter_size)
        self.bias = self.get_initialized_bias(n_filters)
        self.learning_rate = learning_rate
        self.stride = stride
        self.output = None


    def get_initialized_filters(self, nb_filters, filter_size):
        standard_deviation = 1 / np.sqrt(np.prod(filter_size))
        filters = np.empty([nb_filters, filter_size[0], filter_size[1]])
        for i in range(nb_filters):
            filters[i] =  np.random.normal(loc = 0, scale = standard_deviation, size = filter_size)

        return filters


    def get_initialized_bias(self, nb_filters):
        return np.zeros((nb_filters, 1))


    def execute_forward_pass(self, inputs):
        """
        Returns the result of the convolution applied to the inputs passed in parameter.

        Arguments:
            - inputs: 3D input array of shape (input_index, input_height, input_width).

        Returns:
            - 3D array (output_index, output_height, output_width) obtained after performing convolution on input array.
        """
        self.inputs = inputs
        (input_width, input_height) = inputs.shape[1:]

        output_width = int((input_width - self.filter_size[0]) / self.stride) + 1
        output_height = int((input_height - self.filter_size[1]) / self.stride) + 1

        output = np.empty((self.n_filters * inputs.shape[0], output_height, output_width), dtype=int)
        output_x = output_y = 0

        output_index = 0
        for input_array in inputs:
            for filter_index in range(self.n_filters):
                for y in range(0, input_height - self.filter_size[0] + 1, int(self.stride)):
                    for x in range(0, input_width - self.filter_size[1] + 1, int(self.stride)):
                        image_zone = input_array[y:y+self.filter_size[0], x:x+self.filter_size[1]]
                        output[output_index, output_y, output_x] = np.sum(np.multiply(image_zone, self.filters[filter_index]))
                        output_x += 1
                    output_x = 0
                    output_y += 1
                output_y = 0
                output_index += 1

        self.output = relu(output)
        return self.output

    
    def execute_backward_pass(self, d_next_layer):
        """
        Calculates gradient of the cost with respect to the output of this layer and updates the weights.

        Arguments:
            - d_next_layer: gradient of the cost with respect to the output of the next layer.
        
        Returns:
            - the gradient of the cost with respect to the output of this layer.
        """
        d_next_layer = relu_derivative(self.output)
        doutput = np.zeros(self.inputs.shape)
        dfilters = np.zeros(self.filters.shape)

        filter_index = 0
        for input_index in range(self.inputs.shape[0]):
            for y in range(self.filter_size[1]):
                for x in range(self.filter_size[0]):
                    doutput[input_index, y:y+self.filter_size[1], x:x+self.filter_size[0]] += self.filters[filter_index] * d_next_layer[filter_index,y,x]
                    dfilters[filter_index, :, :] += self.output[input_index, y:y+self.filter_size[1], x:x+self.filter_size[0]] * d_next_layer[filter_index,y,x]
            if filter_index < self.n_filters - 1:
                filter_index += 1
            else:
                filter_index = 0

        self.filters += self.learning_rate * dfilters
        return doutput