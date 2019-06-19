import tkinter as tk


def show_failure_popup():
    popup = tk.Tk()
    popup.wm_title("Program completion.")
    label = tk.Label(popup, text="Network failed to optimize architecture.")
    label.pack(side="top", fill="x", pady="10")
    btn = tk.Button(popup, text="Close", command=popup.destroy)
    btn.pack()
    popup.mainloop()