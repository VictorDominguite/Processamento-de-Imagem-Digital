import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

def main():

    img = iio.imread('images/baboon.png')
    #imagem original
    plt.imshow(img, cmap='gray')
    plt.show()

    gamma = 1.5

    for _ in range(3):
        result = (np.rint(((img/255)**(1/gamma))*255)).clip(0,255)
        plt.imshow(result, cmap='gray', vmin = 0, vmax = 255)
        plt.show()
        gamma += 1

main()