import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("images/monarch.pgm", -1)

cv2.imshow("Imagem Original", img)
cv2.waitKey(0)

def get_pixel_value(padded_img, i, j, n, k):
    section = padded_img[i-(n//2):i+(n//2), j-(n//2):j+(n//2)].astype(np.uint32)
    T = np.mean(section) + k * np.std(section)
    if padded_img[i, j] < T:
        return 0
    else:
        return 1

def niblack(img, n , k):
    padded_img = np.pad(img, n, mode='symmetric')
    result = np.empty(img.shape)
    r = 0
    c = 0
    for i in range(n, img.shape[0] + n):
        for j in range(n, img.shape[1] + n):
            result[r, c] = get_pixel_value(padded_img, i, j, n, k)
            c += 1
        c = 0
        r += 1

    cv2.imshow("Metodo de Niblack", result)
    cv2.waitKey(0)
    return result

def plot_histogram(img):
    plt.figure(figsize=(8,6))
    plt.hist(img.flatten(), bins=np.arange(3) - 0.5, edgecolor='black', linewidth=1.4)
    plt.ylabel("Quantidade de pixels")
    plt.xlabel("valor do pixel")
    plt.xticks([0,1])
    plt.title("Distribuição dos pixels na imagem binária")
    plt.show()

result = niblack(img, 15, -0.2)
plot_histogram(result)
print(f"Fracao de pixels pretos: {round(100*((result == 0).sum()/(result.shape[0]*result.shape[1])), 3)}%")