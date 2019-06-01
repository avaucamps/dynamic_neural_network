import numpy as np 
import random
from layers.HiddenLayer import HiddenLayer
from physics_models.attractor import Attractor
from physics_models.particle import Particle
from constants import PHYSICS_WORLD_SIZE
import math

from utils.Vector import Vector, add, sub, divide
from utils.utils import distance_squared, force, distance


class NNArchitectureSeeker:
    def __init__(self, hidden_layers, input_layer, output_layer):
        self.hidden_layers = hidden_layers
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.attractors = []
        self.particles = []
        self._create_attractors_and_particles()


    def _create_attractors_and_particles(self):
        x, y = PHYSICS_WORLD_SIZE
        attractors_distance = (y / len(self.hidden_layers))
        attractors_x = x / 2

        for i in range(1, len(self.hidden_layers) + 1):
            #Gets position for attractors to be evenly spaced
            y = (attractors_distance * i) - (attractors_distance / 2)
            self.attractors.append(Attractor(attractors_x, y, self.hidden_layers[i-1]))
            self._create_particle(attractors_x, y, self.hidden_layers[i-1])


    def _create_particle(self, x, y, hidden_layer):
        radius = 20
        angle_slice = 2 * math.pi / hidden_layer.get_output_shape()
        for i in range(hidden_layer.get_output_shape()):
            angle = angle_slice * i
            x1 = int(x + radius * math.cos(angle))
            y1 = int(y + radius * math.sin(angle))

            self.particles.append(Particle(x1, y1, hidden_layer, i))


    def update_network(self):
        max_distance_part_to_attr = PHYSICS_WORLD_SIZE[1] / len(self.hidden_layers) 

        for i in range(len(self.hidden_layers)):
            attractor = self._get_attractor(self.hidden_layers[i])
            if attractor == None:
                continue

            self._update_neurons(self.hidden_layers[i], attractor, max_distance_part_to_attr)
            if self.output_layer.weights is None:
                return [], ([], [])
            self._update_layer(self.hidden_layers[i], attractor)

        hidden_layers = []
        for layer in self.hidden_layers:
            if layer.weights is not None and layer.weights.size > 0:
                hidden_layers.append(layer)
        self.hidden_layers = hidden_layers

        particles = []
        for particle in self.particles:
            if particle.hidden_layer in self.hidden_layers:
                particles.append(particle)
        self.particles = particles

        attractors = []
        for attractor in self.attractors:
            if attractor.hidden_layer in self.hidden_layers:
                attractors.append(attractor)
        self.attractors = attractors

        self._update_elements()
        self._apply_forces()

        return self.hidden_layers, (self.attractors, self.particles)


    def _get_attractor(self, layer):
        for attractor in self.attractors:
            if attractor.hidden_layer == layer:
                return attractor 

        return None


    def _update_neurons(self, layer, attractor, max_dist):
        for i in range(len(self.particles)):
            if self.output_layer.weights is None:
                return
            if layer.weights is None:
                return

            if self.particles[i].hidden_layer == layer:
                x1, y1 = self.particles[i].position.x, self.particles[i].position.y
                x2, y2 = attractor.position.x, attractor.position.y
                if distance(x1, y1, x2, y2) > max_dist:
                    self._remove_particle(self.particles[i], layer)
                    self._add_particle_to_closest_layer(self.particles[i], max_dist)


    def _update_layer(self, layer, attractor):
        if layer.weights is None or layer.weights.size == 0:
            previous_layer = self._get_previous_layer(layer)
            next_layer = self._get_next_layer(layer)

            next_layer.update_input_shape(previous_layer.get_output_shape())


    def _get_previous_layer(self, layer):
        layer_index = self.hidden_layers.index(layer)
        if layer_index == 0:
            return self.input_layer
        else:
            previous_layer = self.hidden_layers[layer_index - 1]
            while previous_layer.weights is None:
                layer_index -= 1
                previous_layer = self.hidden_layers[layer_index - 1]
                if layer_index == 0:
                    return self.input_layer

            return previous_layer


    def _get_next_layer(self, layer):
        layer_index = self.hidden_layers.index(layer)
        if layer_index == len(self.hidden_layers) - 1:
            return self.output_layer
        else:
            next_layer = self.hidden_layers[layer_index + 1]
            while next_layer.weights is None:
                layer_index += 1
                if layer_index == len(self.hidden_layers) - 1:
                    return self.output_layer
                else:
                    next_layer = self.hidden_layers[layer_index + 1]
                
            
            return next_layer


    def _remove_particle(self, particle, layer):
        #Remove neuron in hidden layer
        layer.remove_neuron(particle.neuron_index)
        layer_index = self.hidden_layers.index(layer)

        #Remove corresponding inputs in next layer
        if layer_index == len(self.hidden_layers) - 1:
            self.output_layer.remove_inputs(particle.neuron_index)
        else:
            self.hidden_layers[layer_index+1].remove_inputs(particle.neuron_index)
        
        self._decrement_neurons_index(layer, particle.neuron_index)

    
    def _decrement_neurons_index(self, layer, start_index):
        for particle in self.particles:
            if particle.hidden_layer == layer:
                if particle.neuron_index > start_index:
                    particle.neuron_index -= 1


    def _add_particle_to_closest_layer(self, particle, max_dist):
        for i in range(len(self.hidden_layers)):
            attractor = self._get_attractor(self.hidden_layers[i])
            if attractor == None:
                continue

            x1, y1 = self.particles[i].position.x, self.particles[i].position.y
            x2, y2 = attractor.position.x, attractor.position.y
            if distance(x1, y1, x2, y2) <= max_dist:
                self.hidden_layers[i].add_neuron()

                #Add input in next layer
                if i == len(self.hidden_layers) - 1:
                    self.output_layer.add_input()
                else:
                    self.hidden_layers[i+1].add_input()
                
                particle.hidden_layer = self.hidden_layers[i]
                return True
        return False


    def _update_elements(self):
        for attractor in self.attractors:
            attractor.update()

        for particle in self.particles:
            particle.update()


    def _apply_forces(self):
        for particle in self.particles:
            particle.attract(self.attractors)
            particle.repulse(self.particles)
            particle.update_vectors()