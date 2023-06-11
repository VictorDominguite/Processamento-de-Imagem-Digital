import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt


def main():
    img = iio.imread('images/araras_RGB.png')

    #imagem original
    plt.imshow(img)
    plt.show()

    # Item A

    result = np.empty(img.shape)

    # altera a dimensao 0 == red
    result[ : , : , 0] = np.rint(np.array(0.393*img[ : , : , 0] + 0.769*img[ : , : , 1] + 0.189*img[ : , : , 2]))
    # altera a dimensao 1 == green
    result[ : , : , 1] = np.rint(np.array(0.349*img[ : , : , 0] + 0.686*img[ : , : , 1] + 0.168*img[ : , : , 2]))
    # altera a dimensao 2 == blue
    result[ : , : , 2] = np.rint(np.array(0.272*img[ : , : , 0] + 0.534*img[ : , : , 1] + 0.131*img[ : , : , 2]))

    result = result.clip(0,255).astype(np.uint8)

    plt.imshow(result)
    plt.show()

    # Item B

    # resultb eh um array com as dimensoes de comprimento e largura da imagem original
    resultb = np.empty((img.shape[0], img.shape[1]))

    resultb[ : , : ] = np.rint(np.array(0.2989*img[ : , : , 0] + 0.5870*img[ : , : , 1] + 0.1140*img[ : , : , 2]))
    resultb.clip(0,255)

    plt.imshow(resultb, cmap='gray', vmin=0, vmax=255)
    plt.show()

main()