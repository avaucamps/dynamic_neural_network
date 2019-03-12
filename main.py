import os
import numpy as np
from data_helper import get_data
from utils import show_digit

X_train, y_train, X_test, y_test = get_data()
show_digit(X_train[0])