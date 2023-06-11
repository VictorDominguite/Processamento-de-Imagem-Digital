import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("images/monarch.pgm", -1)

cv2.imshow("Imagem Original", img)
cv2.waitKey(0)

def metodo_global(img, T):
    
    # Primeira iteração fora do loop para inicializar T_new
    mean1 = (img[img < T]).sum()/(img < T).sum()
    mean2 = (img[img >= T]).sum()/(img >= T).sum()
    T_new = int((mean1 + mean2)/2)

    # Metodo de Ridler e Calvard para encontrar o limiar T
    while (T != T_new):
        T = T_new
        mean1 = (img[img < T]).sum()/(img < T).sum()
        mean2 = (img[img >= T]).sum()/(img >= T).sum()
        T_new = int((mean1 + mean2)/2)
    
    result = np.empty(img.shape)
    result[img<T] = 0
    result[img>=T] = 1
    cv2.imshow("Metodo Global", result)
    cv2.waitKey(0)
    print(f"T = {T}")
    return result

def plot_histogram(img):
    plt.figure(figsize=(8,6))
    plt.hist(img.flatten(), bins=np.arange(3) - 0.5, edgecolor='black', linewidth=1.4)
    plt.ylabel("Quantidade de pixels")
    plt.xlabel("valor do pixel")
    plt.xticks([0,1])
    plt.title("Distribuição dos pixels na imagem binária")
    plt.show()

# Chama o método global com chute inicial do limiar como a media das intensidades da imagem
result = metodo_global(img, np.round(np.mean(img)).astype(int))
plot_histogram(result)
print(f"Fracao de pixels pretos: {round(100*((result == 0).sum()/(result.shape[0]*result.shape[1])), 3)}%")