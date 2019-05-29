'''
Helper class with function used to draw the standard representation of a neural network.
Only the function draw_network_standard_representation should be called as it handles all the drawing.
'''

def draw_network_standard_representation(canvas, hidden_shape, container_width):
    """
    Draws a neural network standard representation in the canvas passed in parameters.

    Args:
    - canvas: tkinter canvas instance.
    - hidden_shape: array containing the number of neurons for each layer.
    - container_width: width of the container to draw the neural network in.

    Returns:
    - the canvas with all the drawings realized.
    """
    canvas, x1 = _draw_input_layer(canvas)
    canvas, x2 = _draw_output_layer(canvas, container_width)
    canvas = _draw_hidden_layers(canvas, hidden_shape, container_width, x1, x2)

    return canvas


def _draw_input_layer(canvas):
    input_layer_end = 90
    canvas.create_text(50, 225, text="Input layer")
    canvas.create_oval(30, 275, 80, 325, fill="#BBB")
    canvas.create_oval(30, 475, 80, 525, fill="#BBB")
    canvas.create_rectangle(20, 250, input_layer_end, 550)

    return canvas, input_layer_end
    

def _draw_output_layer(canvas, container_width):
    output_layer_start = container_width - 90
    canvas.create_text(container_width - 60, 325, text="Output layer")
    canvas.create_oval(container_width - 80, 375, container_width - 30, 425, fill="#BBB")
    canvas.create_rectangle(output_layer_start, 350, container_width - 20, 450)

    return canvas, output_layer_start


def _draw_hidden_layers(canvas, hidden_shape, container_width, x1, x2):
    #text
    canvas.create_text(container_width / 2, 125, text="Hidden layers")

    #draw hidden layers container
    y1 = 150
    y2 = 625
    x1 = x1 + 10
    x2 = x2 - 10
    canvas.create_rectangle(x1, y1, x2, y2)

    #draw the hidden layers
    #divide the width of the container by the number of layers
    #gives space for each layer
    #draw layer in the center of that space
    n_layers = len(hidden_shape)
    layers_container_width = x2 - x1
    layer_container_width = int(layers_container_width / n_layers)

    layer_width = 70
    neuron_radius = 25

    y1 += 10
    y2 -= 10
    layer_container_x1 = x1
    layer_container_x2 = x1 + layer_container_width
    for i in range(n_layers):
        x1_layer = layer_container_x1 + (layer_container_width / 2) - (layer_width / 2)
        x2_layer = layer_container_x1 + (layer_container_width / 2) + (layer_width / 2)
        canvas.create_rectangle(x1_layer, y1, x2_layer, y2)

        #draw neurons
        canvas = _draw_neurons(canvas, hidden_shape[i], layer_container_x1, y1, layer_container_x2, y2, neuron_radius)

        #update coordinates for next layer
        layer_container_x1 = layer_container_x2
        layer_container_x2 += layer_container_width

    return canvas


def _draw_neurons(canvas, n_neurons, layer_x1, layer_y1, layer_x2, layer_y2, neuron_radius):
        #divide the height of the layer by the number of neurons
        #gives space for each neuron
        #draw layer in the center of that space
        layer_container_width = layer_x2 - layer_x1
        neurons_container_height = layer_y2 - layer_y1
        neuron_container_height = int(neurons_container_height / n_neurons)

        neuron_y1 = layer_y1
        neuron_y2 = neuron_y1 + neuron_container_height
        for _ in range(n_neurons):
            x1_neuron = layer_x1 + (layer_container_width / 2) - neuron_radius
            x2_neuron = layer_x1 + (layer_container_width / 2) + neuron_radius
            y1_neuron = neuron_y1 + (neuron_container_height / 2) - neuron_radius
            y2_neuron = neuron_y1 + (neuron_container_height / 2) + neuron_radius
            canvas.create_oval(x1_neuron, y1_neuron, x2_neuron, y2_neuron, fill="#BBB")
            neuron_y1 = neuron_y2
            neuron_y2 += neuron_container_height

        return canvas