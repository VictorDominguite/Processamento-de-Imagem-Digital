import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

def main():
    # Leitura das imagens
    img1 = iio.imread('images/baboon.png')
    img2 = iio.imread('images/butterfly.png')

    # Fatores multiplicativos de cada imagem da composicao
    factor1 = 0.2
    factor2 = 0.8

    # 3 combinacoes diferentes das imagens sao mostradas
    for _ in range(3):
        result = img1 * factor1 + img2 * factor2
        result = np.rint(result)
        factor1 += 0.3
        factor2 -= 0.3
        plt.imshow(result, cmap='gray')
        plt.show()
 
main()