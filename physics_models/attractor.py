from utils.Vector import Vector


class Attractor:
    def __init__(self, x, y, hidden_layer):
        self.position = Vector(x, y)
        self.hidden_layer = hidden_layer
        self.is_alive = True
        self.update()

    
    def update(self):
        if self.is_alive:    
            self.mass = self.hidden_layer.weights.sum()