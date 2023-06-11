import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("images/monarch.pgm", -1)

cv2.imshow("Imagem Original", img)
cv2.waitKey(0)

T, result = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("Metodo de Otsu", result)
cv2.waitKey(0)
print(f'T = {T}')

def plot_histogram(img):
    plt.figure(figsize=(8,6))
    plt.hist(img.flatten(), bins=np.arange(3) - 0.5, edgecolor='black', linewidth=1.4)
    plt.ylabel("Quantidade de pixels")
    plt.xlabel("valor do pixel")
    plt.xticks([0,1])
    plt.title("Distribuição dos pixels na imagem binária")
    plt.show()

result[result==255] = 1
plot_histogram(result)
print(f"Fracao de pixels pretos: {round(100*((result == 0).sum()/(result.shape[0]*result.shape[1])), 3)}%")