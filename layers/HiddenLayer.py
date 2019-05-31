import numpy as np
from .Neuron import Neuron
from utils.utils import truncated_normal

class HiddenLayer:
    def __init__(self, n_neurons, input_shape, activation_function, learning_rate, is_agent_mode_enabled = False):
        self.activation_function = activation_function
        self.is_agent_mode_enabled = is_agent_mode_enabled
        self.input_shape = input_shape
        self.output = np.empty([n_neurons, 1])

        if self.is_agent_mode_enabled:
            self.setup_neurons(n_neurons, input_shape, activation_function)
            self.weights = np.empty([n_neurons, input_shape])
        else:
            self.weights = self.get_init_weights(n_neurons, input_shape)
        
        self.learning_rate = learning_rate


    def get_n_parameters(self):
        if self.is_agent_mode_enabled:
            return self.neurons[0].get_n_parameters() * len(self.neurons)

        return self.weights.shape[0] * self.weights.shape[1]


    def get_output_shape(self):
        return self.weights.shape[0]


    def get_output(self):
        return self.output


    def add_neuron(self):
        if self.is_agent_mode_enabled: return
        new_neuron = self.get_init_weights(1, self.input_shape)
        self.weights = np.concatenate((self.weights, new_neuron), axis=0)


    def add_input(self):
        if self.is_agent_mode_enabled: return
        new_input = self.get_init_weights(self.get_output_shape(), 1)
        self.weights = np.concatenate((self.weights, new_input), axis=1)
        self.input_shape = self.weights.shape[1]


    def remove_neuron(self, index):
        if self.is_agent_mode_enabled: return
        if index > self.get_output_shape(): return

        new_weights_1 = self.weights[:index-1, :]
        new_weights_2 = self.weights[index:, :]
        self.weights = np.concatenate((new_weights_1, new_weights_2), axis=0)

    
    def remove_inputs(self, index):
        if self.is_agent_mode_enabled: return
        if index > self.weights.shape[1]: return

        new_weights_1 = self.weights[:, :index-1]
        new_weights_2 = self.weights[:, index:]
        self.weights = np.concatenate((new_weights_1, new_weights_2), axis=1)
        self.input_shape = self.weights.shape[1]


    #Research subject: study neurons. How to move them ? Do we conserve the weights ?
    def update_n_neurons(self, n_neurons):
        if self.is_agent_mode_enabled: return
        if n_neurons == self.get_output_shape(): return

        new_weights = np.empty([n_neurons, self.input_shape])
        if n_neurons < self.get_output_shape():
            new_weights = self.weights[:n_neurons, :]
        else:
            new_weights[:self.get_output_shape(), :] = self.weights
            new_weights[self.get_output_shape():, :] = self.get_init_weights(
                n_neurons = n_neurons - self.get_output_shape(),
                input_shape = self.input_shape
            )

        self.weights = new_weights


    def get_shape(self):
        return self.weights.shape


    def update_input_shape(self, input_shape):
        if self.is_agent_mode_enabled: return
        if self.input_shape == input_shape: return
        
        new_weights = np.empty([self.get_output_shape(), input_shape])
        if input_shape < self.input_shape:
            new_weights = self.weights[:, :input_shape]
        else:
            new_weights[:, :self.input_shape] = self.weights
            new_weights[:, self.input_shape:] = self.get_init_weights(self.get_output_shape(), input_shape - self.input_shape)

        self.input_shape = input_shape
        self.weights = new_weights


    def setup_neurons(self, n_neurons, input_shape, activation_function):
        self.neurons = []
        for _ in range(n_neurons):
            self.neurons.append(Neuron(input_shape, activation_function))

    
    def get_init_weights(self, n_neurons, input_shape):
        rad = 1 / np.sqrt(input_shape)
        X = truncated_normal(mean=0, 
                             sd=1, 
                             low=-rad, 
                             upp=rad)
        return np.array(X.rvs((n_neurons, input_shape)))

    
    def execute_forward_pass(self, inputs):
        self.inputs = inputs

        if self.is_agent_mode_enabled:
            for i in range(len(self.neurons)):
                self.output[i] = self.neurons[i].run(inputs)
        else:  
            self.output = self.activation_function(np.dot(self.weights, inputs))

        return self.output


    def execute_backward_pass(self, error, next_layer_weights):
        current_layer_error = np.dot(next_layer_weights.T, error)
        gradient = current_layer_error * self.activation_function(self.output, derivative=True)
        gradient = np.dot(gradient, self.inputs.T)

        if self.is_agent_mode_enabled:
            for i in range(len(self.neurons)):
                self.weights[i] = self.neurons[i].update(self.learning_rate * gradient[i])    
        else:
            self.weights -= self.learning_rate * gradient
        
        return self.weights, current_layer_error