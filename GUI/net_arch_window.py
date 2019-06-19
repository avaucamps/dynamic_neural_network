import tkinter as tk

class ChoiceWindow:
    def __init__(self, callback):
        self.callback = callback

    def show_architecture_choice(self):
        self.window = tk.Tk()
        self.window.title("Choose the base architecture")
        self.entries = []
        for i in range(9):
            tk.Label(self.window, text="Number neurons Layer " + str(i+1) + ": ").grid(row=i)
            self.entries.append(tk.Entry(self.window))
            self.entries[i].insert(0, "0")
            self.entries[i].grid(row=i, column=1)
        
        tk.Button(self.window, text="Save", command=self.send_arch).grid(row=9)
        self.window.mainloop()


    def send_arch(self):
        arch = []
        for entry in self.entries:
            if entry.get() != "0":
                arch.append(int(entry.get()))
            else:
                break 

        self.window.destroy()
        self.callback(arch)