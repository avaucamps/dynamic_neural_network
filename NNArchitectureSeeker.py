import numpy as np 
import random


class NNArchitectureSeeker:
    def __init__(self):
        pass


    def search(self, hidden_shape):
        new_hidden_shape = []
        for i in range(random.randint(1,9)):
            new_hidden_shape.append(random.randint(1,9))

        return new_hidden_shape