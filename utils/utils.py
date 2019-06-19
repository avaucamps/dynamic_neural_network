import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
import math


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)


def show_digit(digit):
    plt.imshow(digit, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()


def sigmoid(z, derivative=False):
    if derivative:
        return sigmoid_derivative(z)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1.0 - z)


def relu(x, derivative=False):
    if derivative:
        return relu_derivative(x)
    return np.maximum(0, x)


def relu_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x


def mean_squared_error_derivative(outputs, targets):
    return -(targets - outputs)


def get_categorical_crossentropy_loss(predictions, desired_predictions):
    return -np.sum(desired_predictions * np.log(predictions))


def index_nan_argmax(array):
    idx = np.nanargmax(array)
    idxs = np.unravel_index(idx, array.shape)
    return idxs 


def distance(x1, y1, x2, y2):
    x = x1 - x2
    y = y1 - y2
    return math.sqrt(math.pow(x,2) + math.pow(y, 2))



def distance_squared(x1, y1, x2, y2):
    return distance(x1, y1, x2, y2)

    
def force(m1, m2, d_squared):
    G = 10
    return G * ((m1 * m2) / d_squared)