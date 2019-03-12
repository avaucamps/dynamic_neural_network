from mnist import MNIST
import os
import numpy as np

image_size = 28
n_class = 10

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = MNIST(os.path.join(BASE_PATH, 'data'))


def get_data():
    """
    Retrieves MNIST data and parses it as numpy arrays.

    Returns:
        - X_train: numpy array of shape (number of images, image_size, image_size) with training images.
        - y_train: numpy array of shape (number of labels) with training labels.
        - X_test: numpy array of shape (number of images, image_size, image_size) with test images.
        - y_test: numpy array of shape (number of labels) with test labels.
    """
    images_train, labels_train, images_test, labels_test = load_dataset()
    X_train, y_train, X_test, y_test = prepare_dataset(images_train, labels_train, images_test, labels_test)

    return X_train, y_train, X_test, y_test


def load_dataset():
    images_train, labels_train = DATA_PATH.load_training()
    images_test, labels_test = DATA_PATH.load_testing()

    return images_train, labels_train, images_test, labels_test


def get_labels_vector(labels):
    y = []
    for label in labels:
        output = np.zeros(n_class)
        output[int(label)] = 1
        y.append(output)

    return y


def prepare_dataset(images_train, labels_train, images_test, labels_test):
    X_train = np.asarray(images_train).reshape([len(labels_train), image_size, image_size])
    X_test = np.asarray(images_test).reshape([len(labels_test), image_size, image_size])

    y_train = get_labels_vector(labels_train)
    y_test = get_labels_vector(labels_test)

    return X_train, y_train, X_test, y_test