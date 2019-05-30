from runner import *

# run_mnist_feedforward_network(
#         hidden_shape=[512],
#         learning_rate=0.1,
#         is_agent_mode_enabled=True
# )

# run_cnn(
#         learning_rate=0.01,
#         batch_size=50
# )

run_xor_feedforward_network(False)

# from threading import Thread 
# import tkinter as tk
# from queue import Queue

# class A():
#     def __init__(self, q):
#         self.window = tk.Tk()
#         self.q = q
        
#     def task(self):
#         print(q.get())
#         self.window.after(100, self.task) 

#     def run(self):
#         self.window.after(100, self.task)
#         self.window.mainloop()


# class B(Thread):
#     def __init__(self, q):
#         Thread.__init__(self)
#         self.q = q

#     def run(self):
#         for i in range(10000):
#             self.q.put(i)


# q = Queue()
# a = A(q)
# b = B(q)
# b.start()
# a.run()

#a.join()
#b.join()