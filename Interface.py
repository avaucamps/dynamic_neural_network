import tkinter as tk 
from utils.nn_std_representation_helper import draw_network_standard_representation
from utils.nn_agent_representation_helper import draw_network_agent_representation


class Interface:
    def __init__(self):
        self.window = tk.Tk()
        self.window.wm_title("Neural network dynamic organisation")
        self._setup_canvas()


    def start(self):
        self.window.mainloop()


    def _setup_canvas(self):
        self.width = 1500
        self.height = 800
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, borderwidth=0, highlightthickness=0, bg="white")
        self.canvas.grid()

        self.std_representation_width = 850
        self.agent_representation_width = self.width - self.std_representation_width
        self.canvas.create_line(self.std_representation_width, 0, self.std_representation_width, self.height)
        font = "Arial 13 bold underline"
        self.canvas.create_text(185, 20, font=font, text="Neural network standard representation")
        self.canvas.create_text(1035, 20, font=font, text="Neural network agent representation")
        self.canvas.create_text(910, 70, font="Arial 11", text="Layer: ")
        self.canvas.create_rectangle(950, 63, 964, 77)
        self.canvas.create_text(915, 110, font="Arial 11", text="Neuron: ")
        self.canvas.create_oval(955, 105, 965, 115, fill="#BBB")


    def draw_network_std_representation(self, hidden_shape):
        self.canvas = draw_network_standard_representation(self.canvas, hidden_shape, self.std_representation_width)


    def draw_network_agent_representation(self, hidden_shape):
        x_center = 1300
        self.canvas = draw_network_agent_representation(self.canvas, hidden_shape, x_center, self.height)