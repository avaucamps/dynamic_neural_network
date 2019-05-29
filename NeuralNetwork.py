from utils.utils import sigmoid, relu, relu_derivative, truncated_normal, mean_squared_error_derivative
import numpy as np
from InputLayer import InputLayer
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer
from tqdm import tqdm
from Interface import Interface
import threading


class FullyConnectedNeuralNetwork:
    def __init__(self, input_shape, hidden_shape, output_shape, learning_rate, is_agent_mode_enabled = False):
        """
        Init and builds a fully-connected neural network.

        Arguments:
            - input_shape: number of neurons in input layer.
            - hidden_shape: array with number of neurons for each hidden layer, [int, int ...].
            - output_shape: number of neurons in output layer.
            - learning_rate: learning rate used for optimization.
            - is_agent_mode_enabled: whether or not the neurons should be agents in the hidden layer(s).
        """
        self.interface = Interface()
        self.build_model(input_shape, hidden_shape, output_shape, learning_rate, is_agent_mode_enabled)


    def build_model(self, input_shape, hidden_shape, output_shape, learning_rate, is_agent_mode_enabled):
        self.input_layer = InputLayer()
        self.hidden_layers = []

        last_output_shape = input_shape
        for shape in hidden_shape:
            self.hidden_layers.append(HiddenLayer(shape, last_output_shape, sigmoid, learning_rate, is_agent_mode_enabled))
            last_output_shape = shape

        last_layer_output_shape = self.hidden_layers[-1].get_output_shape()
        self.output_layer = OutputLayer(output_shape, last_layer_output_shape, sigmoid, learning_rate)
        self.interface.draw_network_std_representation(hidden_shape)
        self.interface.draw_network_agent_representation(hidden_shape)
        self.interface.start()


    def train_mnist(self, inputs, outputs):
        for i in tqdm(range(inputs.shape[0])):
            input_sample = inputs[i].reshape([inputs[i].shape[0], 1])
            expected_output = outputs[i].reshape([outputs[i].shape[0], 1])
            forward_pass_output = self.execute_forward_propagation(input_sample)
            error = mean_squared_error_derivative(forward_pass_output, expected_output)
            self.execute_backpropagation(error)
        

    def train_xor(self, inputs, outputs, n_epochs):
        for i in range(n_epochs):
            print("Epoch " + str(i))
            for i in tqdm(range(inputs.shape[0])):
                forward_pass_output = self.execute_forward_propagation(inputs[i].reshape([2,1]))
                error = mean_squared_error_derivative(forward_pass_output, outputs[i])
                self.execute_backpropagation(error)


    def execute_forward_propagation(self, input_vector):
        layer_output = self.input_layer.execute_forward_pass(input_vector)
        for layer in self.hidden_layers:
            layer_output = layer.execute_forward_pass(layer_output)

        layer_output = self.output_layer.execute_forward_pass(layer_output)

        return layer_output

    
    def execute_backpropagation(self, error):
        last_hidden_output = self.hidden_layers[-1].get_output()
        next_layer_weights = self.output_layer.execute_backward_pass(error, last_hidden_output)
        self.hidden_layers.reverse()
        for layer in self.hidden_layers:
            next_layer_weights, error = layer.execute_backward_pass(error, next_layer_weights)
        
        self.hidden_layers.reverse()


    def test(self, input_vector):
        return self.execute_forward_propagation(input_vector)


    def print_n_parameters(self):
        n_parameters = self.output_layer.get_n_parameters()

        for layer in self.hidden_layers:
            n_parameters += layer.get_n_parameters()

        print("{} parameters.".format(str(n_parameters)))