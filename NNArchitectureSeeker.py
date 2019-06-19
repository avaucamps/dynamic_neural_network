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
        self.max_distance_part_to_attr = (PHYSICS_WORLD_SIZE[1] / len(self.hidden_layers)) / 2
        self._create_attractors_and_particles()


    def _create_attractors_and_particles(self):
        x, y = PHYSICS_WORLD_SIZE
        attractors_distance = (y / len(self.hidden_layers))
        attractors_x = x / 2

        for i in range(1, len(self.hidden_layers) + 1):
            #Gets position for attractors to be evenly spaced
            y = (attractors_distance * i) - (attractors_distance / 2)
            self.attractors.append(Attractor(attractors_x, y, self.hidden_layers[i-1]))
            self._create_particles(attractors_x, y, self.hidden_layers[i-1])


    def _create_particles(self, x, y, hidden_layer):
        radius = 20
        angle_slice = 2 * math.pi / hidden_layer.get_output_shape()
        for i in range(hidden_layer.get_output_shape()):
            angle = angle_slice * i
            x1 = int(x + radius * math.cos(angle))
            y1 = int(y + radius * math.sin(angle))

            self.particles.append(Particle(x1, y1, hidden_layer, i))


    def update_network(self, should_add_neuron):
        for i in range(len(self.hidden_layers)):
            if should_add_neuron[i]:
                self._add_neuron(i, self._get_attractor(self.hidden_layers[i]))
        for i in range(len(self.hidden_layers)):
            if not self._is_layer_alive(self.hidden_layers[i]): continue

            attractor = self._get_attractor(self.hidden_layers[i])
            if attractor == None:
                continue

            self._update_neurons(self.hidden_layers[i], attractor)
            if self.output_layer.get_output_shape() == 0:
                return [], ([], [])

        self._remove_dead_particles()
        self._remove_dead_layers()  
        self._update_layers()
        self._update_particles()
        self._update_attractors()

        self._update_elements()
        self._apply_forces()

        if self.output_layer.get_output_shape() == 0:
            return [], ([], [])

        return self.hidden_layers, (self.attractors, self.particles)

    
    def _update_neurons(self, layer, attractor):
        for i in range(len(self.particles)):
            if not self.particles[i].is_alive: continue
            if self.output_layer.weights is None: return

            if self.particles[i].hidden_layer == layer:
                x1, y1 = self.particles[i].position.x, self.particles[i].position.y
                x2, y2 = attractor.position.x, attractor.position.y
                if distance(x1, y1, x2, y2) > self.max_distance_part_to_attr:
                    self.particles[i].is_alive = False      


    def _remove_dead_particles(self):
        for particle in self.particles:
            if not particle.is_alive:
                if not self._is_layer_alive(particle.hidden_layer): continue 

                particle.hidden_layer.remove_neuron(particle.neuron_index)
                self._decrement_neurons_index(particle.hidden_layer, particle.neuron_index)
                next_layer = self._get_next_layer(particle.hidden_layer)
                if next_layer.get_output_shape() != 0:
                    next_layer.remove_inputs(particle.neuron_index)

                self._add_to_closest_layer(particle)


    def _remove_dead_layers(self):
        for layer in self.hidden_layers:
            if not self._is_layer_alive(layer):
                previous_layer = self._get_previous_layer(layer)
                next_layer = self._get_next_layer(layer)
                if next_layer.get_output_shape() != 0:
                    next_layer.update_input_shape(previous_layer.get_output_shape())

                for particle in self.particles:
                    if particle.hidden_layer == layer:
                        particle.is_alive = False


    def _update_layers(self):
        hidden_layers = []
        for layer in self.hidden_layers:
            if self._is_layer_alive(layer):
                hidden_layers.append(layer)
        self.hidden_layers = hidden_layers


    def _update_particles(self):
        particles = []
        for particle in self.particles:
            if particle.is_alive:
                particles.append(particle)
        self.particles = particles


    def _update_attractors(self):
        attractors = []
        for attractor in self.attractors:
            if attractor.hidden_layer in self.hidden_layers:
                attractors.append(attractor)
        self.attractors = attractors


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


    def _get_attractor(self, layer):
        for attractor in self.attractors:
            if attractor.hidden_layer == layer:
                return attractor 

        return None


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


    def _is_layer_alive(self, layer):
        return layer.weights is not None

    
    def _decrement_neurons_index(self, layer, start_index):
        for particle in self.particles:
             if particle.hidden_layer == layer:
                    if particle.neuron_index > start_index:
                        particle.neuron_index -= 1


    def _add_to_closest_layer(self, particle):
        for i in range(len(self.hidden_layers)):
            if not self._is_layer_alive(self.hidden_layers[i]): continue
            attractor = self._get_attractor(self.hidden_layers[i])
            if attractor is None:
                continue

            x1, y1 = particle.position.x, particle.position.y
            x2, y2 = attractor.position.x, attractor.position.y

            if distance(x1, y1, x2, y2) <= self.max_distance_part_to_attr:
                next_layer = self._get_next_layer(self.hidden_layers[i])
                if next_layer.get_output_shape() != 0: 
                    next_layer.add_input()
                    particle.is_alive = True
                    particle.hidden_layer = self.hidden_layers[i]
                    particle.neuron_index = self.hidden_layers[i].get_output_shape()
                    self.hidden_layers[i].add_neuron()
                    return


    def _add_neuron(self, layer_index, attractor):
        x, y = PHYSICS_WORLD_SIZE
        attractors_distance = (y / len(self.hidden_layers))
        attractors_x = x / 2
        y = (attractors_distance * layer_index) - (attractors_distance / 2)

        hidden_layer = self.hidden_layers[layer_index]
        hidden_layer.add_neuron()
        p = Particle(attractors_x + 20, y, hidden_layer, hidden_layer.get_output_shape()-1)
        self.particles.append(p)
        next_layer = self._get_next_layer(hidden_layer)
        next_layer.add_input()