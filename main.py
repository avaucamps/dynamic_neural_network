import os
import numpy as np
from data_helper import get_data
from utils import show_digit
from NeuralNetwork import NeuralNetwork

from utils import relu, sigmoid, softmax

np.random.seed(0)
X_train, y_train, X_test, y_test = get_data()

ANN = NeuralNetwork(
        784, [512], 10, 0.1
)
ANN.train(X_train, y_train)
errors = 0
for i in range(X_test.shape[0]):
    res = ANN.test(X_test[i])
    if np.argmax(res) != np.argmax(np.array(y_test[i])):
        errors += 1

accuracy = 1 - (errors / len(y_test))
print(accuracy)