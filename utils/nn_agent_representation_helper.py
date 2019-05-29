import math

'''
Helper class with function used to draw the agent representation of a neural network.
Only the function draw_network_agent_representation should be called as it handles all the drawing.
'''

def draw_network_agent_representation(canvas, hidden_shape, x_center, height):
    """
    Draws the neural network in the canvas passed in parameter.

    Args:
    - canvas: tkinter canvas instance.
    - hidden_shape: number of neurons for each layer.
    - x_center: center x position for the layer representation.
    - height: height on the canvas.

    Returns:
    - canvas instance with the drawings realized.
    """
    y1 = 0
    y2 = height
    canvas = _draw_agent_layer(canvas, hidden_shape, x_center, y1, y2)

    return canvas


def _draw_agent_layer(canvas, hidden_shape, x_center, y1, y2):
    n_layers = len(hidden_shape)
    container_height = y2 - y1

    layer_container_height = container_height / n_layers
    item_width = 14
    
    layer_container_y1 = y1
    layer_container_y2 = layer_container_y1 + layer_container_height
    for i in range(n_layers):
        y1_layer = layer_container_y1 + (layer_container_height / 2) - (item_width / 2)
        y2_layer = layer_container_y1 + (layer_container_height / 2) + (item_width / 2)
        canvas.create_rectangle(x_center - (item_width / 2), y1_layer, x_center + (item_width / 2), y2_layer)
        
        canvas = _draw_agent_neuron(canvas, hidden_shape[i], x_center, layer_container_y1, layer_container_height, item_width)

        layer_container_y1 = layer_container_y2
        layer_container_y2 += layer_container_height

    return canvas

    
def _draw_agent_neuron(canvas, n_neurons, x_center, layer_container_y1, layer_container_height, item_width):
    radius = 20 + (n_neurons * 2)
    angle_slice = 2 * math.pi / n_neurons
    item_half = item_width / 2
    for i in range(n_neurons):
        angle = angle_slice * i
        x_center_neuron = int(x_center + radius * math.cos(angle))
        y_center_neuron = int((layer_container_y1 + (layer_container_height / 2)) + radius * math.sin(angle))
        canvas.create_oval(x_center_neuron - item_half, y_center_neuron - item_half, x_center_neuron + item_half, y_center_neuron + item_half, fill="#BBB")

    return canvas