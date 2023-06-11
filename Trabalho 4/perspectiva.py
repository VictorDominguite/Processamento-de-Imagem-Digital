import numpy as np
import cv2
import matplotlib.pyplot as plt

# Dada uma matriz de coeficientes de uma transformacao de perspectiva e coordenadas x e y
# sao devolvidos as coordenadas correspondentes x' e y' apos aplicacao da transformacao
def getXY(coef, x, y):
    return  round((coef[0] * x + coef[1] * y + coef[2])/(coef[6] * x + coef[7] * y + 1)), \
            round((coef[3] * x + coef[4] * y + coef[5])/(coef[6] * x + coef[7] * y + 1))


# Dados quatro pontos de origem e seus correspondentes de destino, calcula a matriz de coeficientes para a projecao
# perspectiva
def getPerspectiveCoef(dest, origin):
    # considera a equacao linear matricial ax = b
    # matriz dos coeficientes do sistema linear
    a = np.array([  [-origin[0][0], -origin[0][1], -1, 0, 0, 0, dest[0][0]*origin[0][0], dest[0][0]*origin[0][1] ],
                    [0, 0, 0, -origin[0][0], -origin[0][1], -1, dest[0][1]*origin[0][0], dest[0][1]*origin[0][1] ],
                    
                    [-origin[1][0], -origin[1][1], -1, 0, 0, 0, dest[1][0]*origin[1][0], dest[1][0]*origin[1][1] ],
                    [0, 0, 0, -origin[1][0], -origin[1][1], -1, dest[1][1]*origin[1][0], dest[1][1]*origin[1][1] ],

                    [-origin[2][0], -origin[2][1], -1, 0, 0, 0, dest[2][0]*origin[2][0], dest[2][0]*origin[2][1] ],
                    [0, 0, 0, -origin[2][0], -origin[2][1], -1, dest[2][1]*origin[2][0], dest[2][1]*origin[2][1] ],

                    [-origin[3][0], -origin[3][1], -1, 0, 0, 0, dest[3][0]*origin[3][0], dest[3][0]*origin[3][1] ],
                    [0, 0, 0, -origin[3][0], -origin[3][1], -1, dest[3][1]*origin[3][0], dest[3][1]*origin[3][1] ]
                   ])
    
    b = np.array([  [-dest[0][0]],
                    [-dest[0][1]], 

                    [-dest[1][0]],
                    [-dest[1][1]], 

                    [-dest[2][0]],
                    [-dest[2][1]], 

                    [-dest[3][0]],
                    [-dest[3][1]]
                   ])

    return np.append(np.linalg.solve(a, b).flatten(), 1)

def main():
    img = cv2.imread("images/baboon_perspectiva.png")
    # Sao definidos os 4 pixels de origem e destino correspondentes
    # Considera-se que a origem da transformacao sao as coordenadas
    # da imagem resultante e o destino sao as coordenadas da imagem
    # original, pois estamos buscando na imagem original os pixels
    # que geraram um pixel (x, y) na resultante
    dest = np.array([[37,51], [342, 42], [485, 467], [73, 380]]).astype(np.float32)
    origin = np.array([[0,0], [511, 0], [511, 511], [0, 511]]).astype(np.float32)

    plt.imshow(img, cmap='gray')
    plt.show()
    
    res = np.empty(img.shape)

    # Obtem-se a matriz dos coeficientes da transformacao de perspectiva
    perspective_coef = getPerspectiveCoef(dest, origin)

    # Para cada pixel da imagem resultante, eh calculado qual deveria
    # ser o pixel da imagem original que devia estar naquela posicao
    # para que haja a atribuicao correta dos valores para todas as
    # posicoes da imagem resultante
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x, y = getXY(perspective_coef, j, i)
            res[i,j] = img[y,x]

    plt.imshow(res.astype(np.uint8), cmap='gray')
    plt.show()

main()