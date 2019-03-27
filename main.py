import os
import numpy as np
from data_helper import get_data
from utils import show_digit
from NeuralNetwork import FullyConnectedNeuralNetwork
from CNN import CNN
from utils import relu, sigmoid
from tqdm import tqdm

np.random.seed(0)

def run_feedforward_network():
        X_train, y_train, X_test, y_test = get_data(reshape=True)
        model = FullyConnectedNeuralNetwork(
                input_shape = 784, 
                hidden_shape = [512], 
                output_shape = 10, 
                learning_rate = 0.1,
                is_agent_mode_enabled = False
        )

        print("Epoch 1...")
        model.train(X_train, y_train)
        test_network(model, X_test, y_test)


def run_cnn():
        X_train, y_train, X_test, y_test = get_data()
        model = CNN(n_classes=10, learning_rate=0.01, batch_size=50)
        print("Epoch 1...")
        model.train(X_train[:50], y_train[:50])
        test_network(model, X_test[:5], y_test[:5])


def test_network(model, X_test, y_test):
        errors = 0
        print("Testing...")
        for i in tqdm(range(X_test.shape[0])):
                res = model.test(X_test[i])
                if np.argmax(res) != np.argmax(np.array(y_test[i])):
                        errors += 1

        accuracy = 1 - (errors / len(y_test))
        print("Accuracy is {}".format(accuracy))


#run_feedforward_network()
run_cnn()