import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

def main():
    img = iio.imread('images/baboon.png')

    # mascaras para selecao dos planos de bits
    planos = [0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000, 0b10000000]

    result = np.copy(img)
    for plano in planos:
        result[ : , : ] = img[ : , : ] & plano
        plt.imshow(result, cmap='gray')
        plt.show()

main()