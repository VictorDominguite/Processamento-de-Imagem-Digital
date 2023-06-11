import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import radians

# Mostra a imagem na tela
def show_img(img, cmap='gray', title=""):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Define os argumentos passados para o programa
def add_arguments():
    parser = argparse.ArgumentParser(prog="Transformacoes Geometricas", description="Realiza escala e rotacao de imagens")
    parser.add_argument('-a', metavar='angulo', help="angulo de rotacao medido em graus no sentido anti-horario", type=float, action='store')
    parser.add_argument('-e', metavar='escala', help="fator de escala", type=float, action='store')
    parser.add_argument('-d', metavar=('largura', 'altura'), help="dimensao da imagem de saida em pixels", type=int, action='extend', nargs=2)
    parser.add_argument('-m', metavar='interpolacao', help="metodo de interpolacao utilizado", type=str, action='store', default='bilinear', choices=['vizinho', 'bilinear', 'bicubica', 'lagrange'], dest='metodo')
    parser.add_argument('-i', metavar='imagem', help="imagem de entrada no formato PNG", type=str, action='store', default='images/city.png', dest='source_path')
    parser.add_argument('-o', metavar='imagem', help="imagem de saida no formato PNG (apos a transformacao geometrica)", type=str, action='store', default='results/out.png', dest='out_path')
    return parser

# Limita o valor de um inteiro num entre max_val e min_val
def clip(num, max_val, min_val=0):
    return max(min_val, min(num, max_val))

# Retorna se as coordenadas (x,y) estao dentro de uma image com formato "shape"
def inside_img(x, y, shape):
    return y < shape[0] and x < shape[1]

def P(t):
    return t if t > 0 else 0

def R(s):
    return (1/6)*(P(s+2)**3 - 4*P(s+1)**3 + 6*P(s)**3 - 4*P(s-1)**3)

def L(n, x, y, dx, img, pad_size=4):
    # para ignorar padding, as coordenadas sao acrescidas do tamanho do padding
    x += pad_size
    y += pad_size
    return ((-dx)*(dx-1)*(dx-2)*(img[y+n-2][x-1]))/6 + ((dx+1)*(dx-1)*(dx-2)*(img[y+n-2][x]))/2 + \
           ((-dx)*(dx+1)*(dx-2)*(img[y+n-2][x+1]))/2 + (dx*(dx+1)*(dx-1)*(img[y+n-2][x+2]))/6

# Devolve o valor do pixel com base no metodo e valores calculados para x' e y'
def get_pixel(img, x, y, metodo):
    # int nesse caso atua como a operacao de floor
    dx = x - int(x)
    dy = y - int(y)
    if metodo == 'vizinho':
        newX = clip(round(x), img.shape[1]-1)
        newY = clip(round(y), img.shape[0]-1)
        pixel_value = img[newY][newX]

    elif metodo == 'bilinear':
        pixel_value = (1-dx)*(1-dy)*(img[int(y)][int(x)])
        if (x+1 < img.shape[1]):
            pixel_value += dx*(1-dy)*(img[int(y)][int(x)+1])
        if (y+1 < img.shape[0]):
            pixel_value += (1-dx)*dy*(img[int(y)+1][int(x)])
        if (y+1 < img.shape[0] and x+1 < img.shape[1]):
            pixel_value += dx*dy*(img[int(y)+1][int(x)+1])
    
    elif metodo == 'bicubica':
        pixel_value = 0
        for m in range(-1, 3):
            for n in range(-1, 3):
                if inside_img(int(x)+m, int(y)+n, img.shape):
                    pixel_value += (img[int(y)+n][int(x)+m])*R(m-dx)*R(dy-n)
    
    else:  # lagrange
        pixel_value = ((-dy)*(dy-1)*(dy-2)*L(1, int(x), int(y), dx, img))/6 + ((dy+1)*(dy-1)*(dy-2)*L(2, int(x), int(y), dx, img))/2 + \
                      ((-dy)*(dy+1)*(dy-2)*L(3, int(x), int(y), dx, img))/2 + (dy*(dy+1)*(dy-1)*L(4, int(x), int(y), dx, img))/6
        
    return pixel_value.astype(np.uint8)

# Obtem as coordenadas x' e y' resultantes da rotacao de x e y de um angulo no sentido anti-horario
def get_rotating_XY(angulo, x, y):
    return np.cos(angulo)*x - np.sin(angulo)*y, np.sin(angulo)*x + np.cos(angulo)*y

# Translada as coordenadas x e y de deltaX e deltaY respectivamente
def transladar(x, y, deltaX, deltaY):
    return x + deltaX, y + deltaY

# Rotaciona uma imagem img de um angulo dado no sentido anti-horario seguindo um metodo
# definido para obter o valor dos pixels
def rotacionar(angulo, img, metodo):
    if abs(angulo) == radians(90):
        res = np.empty((img.shape[1], img.shape[0]))
    else:
        res = np.empty(img.shape)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            # translada o centro da imagem para a origem do plano
            j_transladado, i_transladado = transladar(j, i, -res.shape[1]/2, -res.shape[0]/2)
            # rota a imagem
            x, y = get_rotating_XY(angulo, j_transladado, i_transladado)
            # translada a imagem de forma que seu centro esteja no centro da imagem original
            x, y = transladar(x, y, img.shape[1]/2, img.shape[0]/2)
            if x >= img.shape[1] or y >= img.shape[0] or x < 0 or y < 0:
                res[i,j] = np.zeros(res[i,j].shape)
            else:
                if metodo == 'lagrange':
                    res[i,j] = get_pixel(np.pad(img, 4, 'symmetric'), x, y, metodo)
                else:
                    res[i,j] = get_pixel(img, x, y, metodo)
    return res.astype(np.uint8)

# Devolve os valores de x e y com base nos fatores de escala para cada coordenada
def get_scaling_XY(Sx, Sy, x, y):
    return Sx*x, Sy*y

# Escala uma imagem com base em um fator de escala ou em dimensoes dadas para a imagem
def escalar(args, img, metodo):
    if args.e:
        # a escala eh feita com fator de escala
        res = np.zeros((round(args.e*img.shape[0]), round(args.e*img.shape[1])))
        Sx = Sy = 1/args.e
    else:
        # a escala eh feita com dimensoes
        Sx = img.shape[1]/args.d[0]
        Sy = img.shape[0]/args.d[1]
        res = np.zeros((args.d[1], args.d[0]))
    
    for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                x, y = get_scaling_XY(Sx, Sy, j, i)
                if metodo == 'lagrange':
                    res[i,j] = get_pixel(np.pad(img, 4, 'symmetric'), x, y, metodo)
                else:
                    res[i,j] = get_pixel(img, x, y, metodo)
        
    return res.astype(np.uint8)

def main():
    parser = add_arguments()
    args = parser.parse_args()
    img = cv2.imread(args.source_path)
    img = img[:,:,0]

    show_img(img, title="Imagem original")
    
    if args.a is not None and (args.e is not None or args.d is not None):
        print("Nao e possivel realizar escala e rotacao na mesma execucao, apenas uma operacao pode ser realizada por vez")
        return 0
    if args.e and args.d:
        print("Especifique apenas um dos criterios de redimensionamento (fator de escala ou dimensoes)")
        return 0
    if args.a:
        res = rotacionar(radians(args.a), img, args.metodo)
    elif args.e or args.d:
        res = escalar(args, img, args.metodo)

    else:
        print("Alguma operacao (rotacao ou escala) deve ser especificada")
    
    show_img(res, title="Imagem resultante")
    plt.imsave(args.out_path, res, cmap='gray')
    # show_img(res[3*int(res.shape[0]/4)-100:3*int(res.shape[0]/4)+300, int(res.shape[1]/2)-200:int(res.shape[1]/2)+200], title="Imagem resultante")
    # plt.imsave(args.out_path[:-4]+'_zoom.png', res[3*int(res.shape[0]/4)-100:3*int(res.shape[0]/4)+300, int(res.shape[1]/2)-200:int(res.shape[1]/2)+200], cmap='gray')


main()