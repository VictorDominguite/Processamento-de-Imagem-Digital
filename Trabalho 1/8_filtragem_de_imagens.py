import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

def get_filters():
    h1 = np.array([
        [ 0,  0, -1,  0,  0],
        [ 0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [ 0, -1, -2, -2,  0],
        [ 0,  0, -1,  0,  0]
    ])

    h2 = (1/256) * np.array([
        [1,  4,  6,  4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1,  4,  6,  4, 1]
    ])

    h3 = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    h4 = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ])

    h5 = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])

    h6 = (1/9) * np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    h7 = np.array([
        [-1, -1, 2],
        [-1, 2, -1],
        [2, -1, -1]
    ])

    h8 = np.array([
        [2, -1, -1],
        [-1, 2, -1],
        [-1, -1, 2]
    ])

    h9 = (1/9) * np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])

    h10 = (1/8) * np.array([
        [-1, -1, -1, -1, -1],
        [-1,  2,  2,  2, -1],
        [-1,  2,  8,  2, -1],
        [-1,  2,  2,  2, -1],
        [-1, -1, -1, -1, -1]
    ])

    h11 = np.array([
        [-1, -1,  0],
        [-1,  0,  1],
        [ 0,  1,  1]
    ])
    return [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11]

def calculate_pixel(mask: np.array, i, j, img: np.array):
    '''
    Devolve o valor de um pixel de uma imagem apos a aplicacao de uma mascara
    '''
    pixel = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            pixel += mask[r, c] * img[i+r, j+c]
    return round(pixel)

def apply_filter(altered_image: np.array, img_shape: np.shape, mask: np.array):
    '''
    A funcao recebe uma imagem alterada (com bordas para a aplicacao da mascara),
    o shape (altura e largura) da imagem resultante e uma mascara para aplicacao
    do filtro. Devolve uma imagem aplicando o filtro por correlação
    '''
    result = np.empty((img_shape[0], img_shape[1]))
    for r in range(img_shape[0]):
        for c in range(img_shape[1]):
            result[r, c] = calculate_pixel(mask, r, c, altered_image)
    return np.clip(result, 0, 255)

def main():

    img = iio.imread('images/baboon.png')

    filters = get_filters()

    # filtered eh um vetor que contera as 12 imagens filtradas
    filtered = np.empty((12, img.shape[0], img.shape[1]))

    # Aplicacao dos 11 primeiros filtros
    for i in range(len(filters)):
        filtered[i] = apply_filter(np.pad(img, filters[i].shape[0]//2, 'symmetric'), img.shape, filters[i])
        plt.imshow(filtered[i], cmap='gray', vmin=0, vmax=255)
        plt.show()

    # Combinacao dos filtros 3 e 4
    filtered[11] = np.rint(np.sqrt(np.square(filtered[2]) + np.square(filtered[3])))
    plt.imshow(filtered[11], cmap='gray', vmin=0, vmax=255)
    plt.show()

main()