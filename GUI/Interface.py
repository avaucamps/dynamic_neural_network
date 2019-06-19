import tkinter as tk
import types
from queue import Queue
from .StdRepresentationHelper import StdRepresentationHelper
from .AgentRepresentationHelper import AgentRepresentationHelper
from DisplayData import DisplayData


class Interface():
    def __init__(self, queue):
        self.queue = queue
        self.window = tk.Tk()
        self.window.wm_title("Neural network dynamic organisation")
        self._setup_canvas()
        self.std_representation_helper = StdRepresentationHelper(self.canvas)
        self.agent_representation_helper = AgentRepresentationHelper(self.canvas, self.middle, 0, self.width-self.middle, self.height)


    def run(self):
        self.window.after(100, self._update)
        self.window.mainloop()


    def _setup_canvas(self):
        self.width = 1920
        self.height = 1080
        self.middle = (1920/2)
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, borderwidth=0, highlightthickness=0, bg="white")
        self.canvas.grid()

        self.std_representation_width = self.width / 2
        self.agent_representation_width = self.width - self.std_representation_width
        self.canvas.create_line(self.std_representation_width, 0, self.std_representation_width, self.height)
        font = "Arial 13 bold underline"
        self.canvas.create_text(185, 20, font=font, text="Neural network standard representation")
        self.canvas.create_text(1145, 20, font=font, text="Neural network agent representation")
        self.canvas.create_text(1000, 70, font="Arial 11", text="Layer: ")
        self.canvas.create_rectangle(1025, 65, 1035, 75)
        self.canvas.create_text(995, 110, font="Arial 11", text="Neuron: ")
        self.canvas.create_oval(1025, 105, 1035, 115, fill="#BBB")


    def _draw_network_std_representation(self, hidden_shape):
        self.std_representation_helper.draw_representation(
            hidden_shape=hidden_shape, 
            container_width=self.std_representation_width
        )


    def _draw_network_agent_representation(self, attractors, particles):
        self.agent_representation_helper.draw_physics_modelisation(attractors, particles)


    def _update(self):
        message = self.queue.get()
        if message:
            if isinstance(message, DisplayData):
                self._draw_network_std_representation(message.hidden_shape)
                self._draw_network_agent_representation(message.attractors, message.particles)
            elif isinstance(message, str) and message == "Done":
                self.window.destroy()
                return

        self.canvas.update()
        self.window.after(100, self._update)