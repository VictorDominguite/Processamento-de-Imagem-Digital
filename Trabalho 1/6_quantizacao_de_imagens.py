import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

def main():
    img = iio.imread('images/baboon.png')
    niveis = [256, 64, 32, 16, 8, 4, 2]
    max_pixel = np.max(img)

    for nvl in niveis:
        img_quantizada = (np.rint((nvl-1)*(img/max_pixel))* (255/(nvl-1))).clip(0,255).astype(np.uint8)
        plt.imshow(img_quantizada, cmap='gray')
        plt.show()

main()