from utils.utils import sigmoid, relu, relu_derivative, truncated_normal, mean_squared_error_derivative
import numpy as np
from layers.InputLayer import InputLayer
from layers.HiddenLayer import HiddenLayer
from layers.OutputLayer import OutputLayer
from tqdm import tqdm
from GUI.Interface import Interface
from NNArchitectureSeeker import NNArchitectureSeeker
from threading import Thread
from DisplayData import DisplayData
import time


class FullyConnectedNeuralNetwork(Thread):
    def __init__(self, queue, input_shape, hidden_shape, output_shape, learning_rate, is_agent_mode_enabled = False):
        """
        Init and builds a fully-connected neural network.

        Arguments:
            - queue: queue object to put messages into to send to interface.
            - input_shape: number of neurons in input layer.
            - hidden_shape: array with number of neurons for each hidden layer, [int, int ...].
            - output_shape: number of neurons in output layer.
            - learning_rate: learning rate used for optimization.
            - is_agent_mode_enabled: whether or not the neurons should be agents in the hidden layer(s).
        """
        Thread.__init__(self)
        self.queue = queue
        self._build_model(input_shape, hidden_shape, output_shape, learning_rate, is_agent_mode_enabled)


    def set_data(self, inputs, outputs, n_epochs, network_to_run):
        self.inputs = inputs
        self.outputs = outputs
        self.n_epochs = n_epochs
        self.network_to_run = network_to_run


    def run(self):
        if self.network_to_run == "xor":
            self._train_xor()
        elif self.network_to_run == "mnist":
            self._train_mnist()
        else:
            print("ERROR: network to run not supported.")


    def _build_model(self, input_shape, hidden_shape, output_shape, learning_rate, is_agent_mode_enabled):
        self.input_layer = InputLayer(input_shape)
        self.hidden_layers = []

        last_output_shape = input_shape
        for shape in hidden_shape:
            self.hidden_layers.append(HiddenLayer(shape, last_output_shape, sigmoid, learning_rate, is_agent_mode_enabled))
            last_output_shape = shape

        last_layer_output_shape = self.hidden_layers[-1].get_output_shape()
        self.output_layer = OutputLayer(output_shape, last_layer_output_shape, sigmoid, learning_rate)
        self.architecture_seeker = NNArchitectureSeeker(self.hidden_layers, self.input_layer, self.output_layer)


    def _train_xor(self):
        self.queue.put(self._get_hidden_shape())
        for i in range(self.n_epochs):
            print("Epoch " + str(i))
            for i in tqdm(range(self.inputs.shape[0])):
                forward_pass_output = self._execute_forward_propagation(self.inputs[i].reshape([2,1]))
                error = mean_squared_error_derivative(forward_pass_output, self.outputs[i])
                self._execute_backpropagation(error)
            
            self.hidden_layers, (attractors, particles) = self.architecture_seeker.update_network()
            self.queue.put(DisplayData(self._get_hidden_shape(), attractors, particles))
            time.sleep(0.05)
        self.queue.put("Done")


    def _train_mnist(self):
        for i in range(self.n_epochs):
            print("Epoch " + str(i))
            for i in tqdm(range(self.inputs.shape[0])):
                input_sample = self.inputs[i].reshape([self.inputs[i].shape[0], 1])
                expected_output = self.outputs[i].reshape([self.outputs[i].shape[0], 1])
                forward_pass_output = self._execute_forward_propagation(input_sample)
                error = mean_squared_error_derivative(forward_pass_output, expected_output)
                self._execute_backpropagation(error)


    def _execute_forward_propagation(self, input_vector):
        layer_output = self.input_layer.execute_forward_pass(input_vector)
        for layer in self.hidden_layers:
            layer_output = layer.execute_forward_pass(layer_output)

        layer_output = self.output_layer.execute_forward_pass(layer_output)

        return layer_output

    
    def _execute_backpropagation(self, error):
        last_hidden_output = self.hidden_layers[-1].get_output()
        next_layer_weights = self.output_layer.execute_backward_pass(error, last_hidden_output)
        self.hidden_layers.reverse()
        for layer in self.hidden_layers:
            next_layer_weights, error = layer.execute_backward_pass(error, next_layer_weights)
        
        self.hidden_layers.reverse()


    def _get_hidden_shape(self):
        hidden_shape = []
        for i in range(len(self.hidden_layers)):
            hidden_shape.append(self.hidden_layers[i].get_output_shape())
        
        return hidden_shape


    def test(self, input_vector):
        return self._execute_forward_propagation(input_vector)


    def _print_n_parameters(self):
        n_parameters = self.output_layer.get_n_parameters()

        for layer in self.hidden_layers:
            n_parameters += layer.get_n_parameters()

        print("{} parameters.".format(str(n_parameters)))