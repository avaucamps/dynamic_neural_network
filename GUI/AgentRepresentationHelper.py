import math


class AgentRepresentationHelper:
    '''
    Helper class with function used to draw the agent representation of a neural network.
    '''
    def __init__(self, canvas):
        self.canvas = canvas
        self.removable_tag = "removable"


    def draw_representation(self, hidden_shape, x_center, height):
        """
        Draws the neural network in the canvas passed in parameter.

        Args:
        - hidden_shape: number of neurons for each layer.
        - x_center: center x position for the layer representation.
        - height: height on the canvas.
        """
        y1 = 0
        y2 = height
        self._draw_agent_layer(hidden_shape, x_center, y1, y2)


    def _delete_items(self):
        self.canvas.delete(self.removable_tag)


    def _draw_agent_layer(self, hidden_shape, x_center, y1, y2):
        n_layers = len(hidden_shape)
        container_height = y2 - y1

        layer_container_height = container_height / n_layers
        item_width = 14
        
        layer_container_y1 = y1
        layer_container_y2 = layer_container_y1 + layer_container_height
        for i in range(n_layers):
            y1_layer = layer_container_y1 + (layer_container_height / 2) - (item_width / 2)
            y2_layer = layer_container_y1 + (layer_container_height / 2) + (item_width / 2)
            self.canvas.create_rectangle(x_center - (item_width / 2), y1_layer, x_center + (item_width / 2), y2_layer, tags=self.removable_tag)
            
            self._draw_agent_neuron(hidden_shape[i], x_center, layer_container_y1, layer_container_height, item_width)

            layer_container_y1 = layer_container_y2
            layer_container_y2 += layer_container_height

        
    def _draw_agent_neuron(self, n_neurons, x_center, layer_container_y1, layer_container_height, item_width):
        radius = 20 + (n_neurons * 2)
        angle_slice = 2 * math.pi / n_neurons
        item_half = item_width / 2
        for i in range(n_neurons):
            angle = angle_slice * i
            x_center_neuron = int(x_center + radius * math.cos(angle))
            y_center_neuron = int((layer_container_y1 + (layer_container_height / 2)) + radius * math.sin(angle))
            x1 = x_center_neuron - item_half 
            y1 = y_center_neuron - item_half
            x2 = x_center_neuron + item_half 
            y2 = y_center_neuron + item_half
            self.canvas.create_oval(x1, y1, x2, y2, fill="#BBB", tags=self.removable_tag)