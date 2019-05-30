import tkinter as tk
import types
from queue import Queue
from .StdRepresentationHelper import StdRepresentationHelper
from .AgentRepresentationHelper import AgentRepresentationHelper


class Interface():
    def __init__(self, queue):
        self.queue = queue
        self.window = tk.Tk()
        self.window.wm_title("Neural network dynamic organisation")
        self._setup_canvas()
        self.std_representation_helper = StdRepresentationHelper(self.canvas)
        self.agent_representation_helper = AgentRepresentationHelper(self.canvas)


    def run(self):
        self.window.after(100, self._update)
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


    def _draw_network_std_representation(self, hidden_shape):
        self.std_representation_helper.draw_network_standard_representation(
            hidden_shape=hidden_shape, 
            container_width=self.std_representation_width
        )


    def _draw_network_agent_representation(self, hidden_shape):
        x_center = 1300
        self.agent_representation_helper.draw_network_agent_representation(
            hidden_shape=hidden_shape, 
            x_center=x_center, 
            height=self.height
        )


    def _update(self):
        hidden_shape = self.queue.get()
        if hidden_shape and isinstance(hidden_shape, list):
            self._draw_network_std_representation(hidden_shape)
            self._draw_network_agent_representation(hidden_shape)

        self.canvas.update()
        self.window.after(100, self._update)