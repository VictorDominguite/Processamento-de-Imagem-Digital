import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

def main():
    # Leitura da imagem
    img = iio.imread('images/city.png')

    # Negativo da imagem
    negative = 255 - np.copy(img)
    plt.imshow(negative, cmap='gray')
    plt.show()

    # Transformacao no intervalo de intensidades
    transformed = np.array(100*(img / 255) + 100, dtype = 'uint8')
    plt.imshow(transformed, cmap='gray', vmin=0, vmax=255)
    plt.show()

    # Linhas pares invertidas
    par_invertido = np.copy(img)
    # percorre a matriz da linha 0 até o fim de 2 em 2 e troca pela própria linha invertida
    par_invertido[0::2, : ] = par_invertido[0::2, ::-1]
    plt.imshow(par_invertido, cmap='gray')
    plt.show()

    # Reflexao de linhas
    refletida = np.copy(img)
    refletida[refletida.shape[0]//2: , : ] = refletida[refletida.shape[0]//2:0:-1, : ]
    plt.imshow(refletida, cmap='gray')
    plt.show()

    # Espelhamento vertical
    espelhada = np.copy(img)
    espelhada[ : , : ] = espelhada[ ::-1 , : ]
    plt.imshow(espelhada, cmap='gray')
    plt.show()
 
main()