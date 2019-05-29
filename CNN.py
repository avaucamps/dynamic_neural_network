import numpy as np
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer
from ConvolutionLayer import ConvolutionLayer
from MaxPoolLayer import MaxPoolLayer
from FlattenLayer import FlattenLayer
from utils.utils import relu, sigmoid, mean_squared_error_derivative, show_digit
from tqdm import tqdm

class CNN:
    def __init__(self, n_classes, learning_rate, batch_size=1):
        """
        Init and builds a convolutional neural network.

        Arguments:
            - n_classes: number of classes to predict.
            - learning_rate: learning rate used for optmimization.
        """
        self.build_model(n_classes, learning_rate)
        self.batch_size = batch_size
        self.n_classes = n_classes

    
    def build_model(self, n_classes, learning_rate):
        self.model = []
        self.model.append(ConvolutionLayer(n_filters=32, filter_size=(3,3), learning_rate=learning_rate))
        self.model.append(ConvolutionLayer(n_filters=64, filter_size=(3,3), learning_rate=learning_rate))
        self.model.append(MaxPoolLayer(window_size=(2, 2), stride=2))
        self.model.append(FlattenLayer())
        self.model.append(HiddenLayer(n_neurons=128, input_shape=294912, activation_function=relu, learning_rate=learning_rate))
        self.model.append(OutputLayer(n_classes, input_shape=128, activation_function=sigmoid, learning_rate=learning_rate))


    def train(self, inputs, outputs):
        for i in tqdm(range(inputs.shape[0])):
            input_sample = inputs[i].reshape([1, inputs.shape[1], inputs.shape[2]])
            expected_output = outputs[i].reshape([outputs[i].shape[0], 1])
            forward_pass_output = self.execute_forward_propagation(input_sample)
            error = mean_squared_error_derivative(forward_pass_output, expected_output)
            self.execute_backpropagation(error)


    def execute_forward_propagation(self, input_array):
        output_array = input_array
        for layer in self.model:
            output_array = layer.execute_forward_pass(output_array)

        return output_array
    

    def execute_backpropagation(self, error):
        last_hidden_output = self.model[-2].get_output()
        next_layer_weights = self.model[-1].execute_backward_pass(error, last_hidden_output)
        next_layer_weights, error = self.model[-2].execute_backward_pass(error, next_layer_weights)
        d_pool = self.model[-3].execute_backward_pass(error, next_layer_weights)
        d_conv2 = self.model[-4].execute_backward_pass(d_pool)
        d_conv1 = self.model[-5].execute_backward_pass(d_conv2)
        self.model[-6].execute_backward_pass(d_conv1)


    def test(self, input_vector):
        input_vector = input_vector.reshape([1, input_vector.shape[0], input_vector.shape[1]])
        return self.execute_forward_propagation(input_vector)