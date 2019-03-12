import matplotlib.pyplot as plt

def show_digit(digit):
    plt.imshow(digit, cmap=plt.cm.gray_r, interpolation="nearest")
    plt.show()