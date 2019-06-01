from utils.Vector import Vector, add, sub, divide
from utils.utils import distance_squared, force


class Particle:
    def __init__(self, x, y, hidden_layer, neuron_index):
        self.position = Vector(x, y)
        self.velocity = Vector(0, 0)
        self.acceleration = Vector(0, 0)
        self.hidden_layer = hidden_layer
        self.neuron_index = neuron_index
        self.is_alive = True


    def update(self):
        if self.is_alive:
            self.mass =  self.hidden_layer.weights[self.neuron_index, :].sum()
            self.acceleration = Vector(0,0)


    def update_vectors(self):
        if self.is_alive:
            self.velocity = add(self.velocity, self.acceleration)
            self.position = add(self.position, self.velocity)


    def attract(self, attractors):
        if self.is_alive:
            for attractor in attractors:
                if not attractor.is_alive: continue

                direction = sub(attractor.position, self.position)
                dist = distance_squared(
                    attractor.position.x,
                    attractor.position.y,
                    self.position.x,
                    self.position.y
                )
                magnitude = force(attractor.mass, self.mass, dist)
                direction.set_magnitude(magnitude)
                self.acceleration = add(self.acceleration, divide(direction, self.mass))


    def repulse(self, repulsors):
        if self.is_alive:
            for repulsor in repulsors:
                if repulsor == self: continue
                if not repulsor.is_alive: continue

                direction = sub(repulsor.position, self.position)
                dist = distance_squared(
                    repulsor.position.x,
                    repulsor.position.y,
                    self.position.x,
                    self.position.y
                )

                if dist < 25:
                    dist = 25
                elif dist > 100:
                    dist = 100

                magnitude = force(repulsor.mass, self.mass, dist) * -1
                direction.set_magnitude(magnitude)
                self.acceleration = add(self.acceleration, divide(direction, self.mass))