import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

def main():
    # Leitura das imagens
    img = iio.imread('images/baboon.png')

    height, width = img.shape
    # new_h e new_w representam as dimensoes (altura e largura) de cada bloco do mosaico
    new_h, new_w = height//4, width//4

    # Cada bloco na imagem original eh reposicionado a seguir para compor o mosaico
    mosaico = np.block([
    [img[new_h:2*new_h,   new_w:2*new_w  ],   img[2*new_h:3*new_h, 2*new_w:3*new_w],  img[3*new_h:4*new_h,       0:new_w], img[0:new_h,        2*new_w:3*new_w]],
    [img[new_h:2*new_h,   3*new_w:4*new_w],   img[3*new_h:4*new_h, 3*new_w:4*new_w],  img[0:new_h,               0:new_w], img[2*new_h:3*new_h ,     0:new_w  ]],
    [img[2*new_h:3*new_h, 3*new_w:4*new_w],   img[3*new_h:4*new_h, new_w:2*new_w  ],  img[0:new_h,         new_w:2*new_w], img[new_h:2*new_h , 2*new_w:3*new_w]],
    [img[0:new_h,         3*new_w:4*new_w],   img[3*new_h:4*new_h, 2*new_w:3*new_w],  img[2*new_w:3*new_h, new_w:2*new_w], img[new_h:2*new_h ,       0:new_w  ]]
    ])

    plt.imshow(mosaico, cmap='gray')
    plt.show()

main()