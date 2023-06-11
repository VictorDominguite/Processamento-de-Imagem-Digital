import numpy as np
import cv2
from matplotlib import pyplot as plt

class Rect():
    def __init__(self, x, y, w, h) -> None:
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def show_img(img, cmap='gray', vmin=0, vmax=1, title=""):
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_colored_labels_img(labels_img):
    # adaptado de https://iq.opengenus.org/connected-component-labeling/
    # Cria imagem no formato HSV
    hue = np.uint8(179*labels_img/np.max(labels_img))
    ones = 255*np.ones_like(hue)
    colored = cv2.merge([hue, ones, ones])
    # Converte para BGR
    colored = cv2.cvtColor(colored, cv2.COLOR_HSV2BGR)
    # Define o fundo como preto
    colored[hue==0] = 0
    plt.imshow(colored)
    plt.title("Componentes conexos separados por cor")
    plt.show()

def removeRectsInsideOtherRect(rectangles):
    idxs = np.array([])
    for i in range(len(rectangles)):
        r = rectangles[i]
        for rec in rectangles:
            if rec != r and (r.x >= rec.x and r.y >= rec.y) and (r.x + r.w <= rec.x + rec.w and r.y + r.h <= rec.y + rec.h):
                idxs = np.append(idxs, i).astype(int)
                break
    if np.any(idxs):
        rectangles = np.delete(rectangles, idxs)
    return rectangles

# Devolve uma imagem com os retangulos desenhados em torno dos objetos e uma lista
# de retangulos
def boundingRectangles(img):
    contours, _ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    rectangles = np.array([])
    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        rectangles = np.append(rectangles, Rect(x,y,w,h))

    rectangles = removeRectsInsideOtherRect(rectangles)
    
    for r in rectangles:
        cv2.rectangle(img,(r.x,r.y),(r.x+r.w,r.y+r.h),color=4,thickness=3)

    return img, rectangles

# Devolve uma lista de porcentagens de pixels pretos para cada retangulo de rectangles
# na imagem img, as porcentagens estao na mesma ordem que os retangulos originais
def getPixelsProportion(rectangles, img):
    proportions = np.array([])
    for r in rectangles:
        black_pixels = 0
        for i in range(r.y, r.y + r.h):
            for j in range(r.x, r.x + r.w):
                black_pixels += not img[i, j]

        proportions = np.append(proportions, black_pixels/(r.w * r.h))
    return proportions

# Devolve uma lista em que cada elemento corresponde a soma das transicoes entre pixels
# brancos e pretos no horizontal e vertical dividido pelo numero de pixels pretos para 
# cada retângulo
def getTransitionsProportion(rectangles, img):
    proportions = np.array([])
    for r in rectangles:
        black_pixels = 0
        transitions = 0

        for i in range(r.y, r.y + r.h):
            for j in range(r.x, r.x + r.w):
                #TODO 
                # transicoes verticais
                if i > 0 and img[i, j] == 0 and img[i-1, j] == 1:
                    transitions += 1
                # transicoes horizontais
                if j > 0 and img[i, j] == 0 and img[i, j-1] == 1:
                    transitions += 1

                black_pixels += not img[i, j]
        if black_pixels == 0:
            proportions = np.append(proportions, 0)
        else:
            proportions = np.append(proportions, transitions/black_pixels)

    return proportions


# Devolve o numero de elementos reconhecidos como texto e uma lista de retangulos que
# envolvem cada um desses elementos
def getText(black_pixels_percentages, transitions_proportions, rectangles, getWords=False):
    text = np.array([])
    num_text = 0
    black_pixels_constants = [0.2, 0.4]
    transitions_constants = [0.3, 0.38]
    if getWords:
        black_pixels_constants = [0.14, 0.67]
        transitions_constants = [0.28, 0.47]
    for i in range(len(rectangles)):
        if black_pixels_percentages[i] > black_pixels_constants[0] and black_pixels_percentages[i] < black_pixels_constants[1] and \
            transitions_proportions[i] > transitions_constants[0] and transitions_proportions[i] < transitions_constants[1]:
            text = np.append(text, rectangles[i])
            num_text += 1
    return num_text, text

# Mostra a imagem img com os retangulos em text_rect desenhados em cima dela
def draw_text_rects(img, text_rects, title=""):
    img_cpy = np.copy(img)
    for r in text_rects:
        cv2.rectangle(img_cpy,(r.x,r.y),(r.x+r.w,r.y+r.h),color=0,thickness=3)
    show_img(img_cpy, vmax=np.max(img_cpy), title=title)

# Realiza operacoes para obter o numero de palavras no texto
def getWords(img, img_aux):
    elem_estr = np.ones((1, 11), np.uint8)
    words_prep = cv2.dilate(img_aux, elem_estr, iterations=1) # dilata para unir letras de uma mesma palavra
    show_img(words_prep, title="Detectando palavras")

    bounding_rec_img_words, rectangles_words = boundingRectangles(words_prep)   # Encontra os retangulos em torno de componentes conexas
    show_img(bounding_rec_img_words, vmax=np.max(bounding_rec_img_words), title="Retângulos ao redor das palavras")

    black_pixels_percentages_words = getPixelsProportion(rectangles_words, img)

    transitions_proportions_words = getTransitionsProportion(rectangles_words, img)

    # Define quais componentes conexas podem ser classificados como texto e imprime os resultados
    num_words, text_rects_words = getText(black_pixels_percentages_words, transitions_proportions_words, rectangles_words, getWords=True)
    draw_text_rects(img, text_rects_words, title="Objetos identificados como texto")
    
    print(f"Ha aproximadamente {num_words} palavras no texto\n")


def main():

    img = (cv2.imread("bitmap.pbm", -1) / 255).astype(np.uint8)

    # Imagem invertida para os objetos serem representados por 1s
    # e assim poder fazer as opercoes morfologicas
    img_aux = np.logical_not(img).astype(np.uint8)
    show_img(img_aux, title="Imagem invertida para realizar operacoes")

    elem_estr = np.ones((1, 100), np.uint8)
    res1 = cv2.dilate(img_aux, elem_estr, iterations=1).astype(np.uint8)
    show_img(res1, title="Resultado passo 1")

    res2 = cv2.erode(res1, elem_estr, iterations=1).astype(np.uint8)
    show_img(res2, title="Resultado passo 2")

    elem_estr = np.ones((200, 1), np.uint8)
    res3 = cv2.dilate(img_aux, elem_estr, iterations=1).astype(np.uint8)
    show_img(res3, title="Resultado passo 3")

    res4 = cv2.erode(res3, elem_estr, iterations=1).astype(np.uint8)
    show_img(res4, title="Resultado passo 4")

    res5 = np.logical_and(res2, res4).astype(np.uint8).astype(np.uint8)
    show_img(res5, title="Resultado passo 5")

    elem_estr = np.ones((1, 30), np.uint8)
    res6 = cv2.morphologyEx(res5, cv2.MORPH_CLOSE,  elem_estr, iterations=1).astype(np.uint8)
    show_img(res6, title="Resultado passo 6")

    num_labels, labels_img = cv2.connectedComponents(res6)
    show_colored_labels_img(labels_img)

    # Obtencao de retangulos que contornam os objetos
    bounding_rec_img, rectangles = boundingRectangles(res6)
    show_img(bounding_rec_img, vmax=np.max(bounding_rec_img), title="Retângulos ao redor dos objetos")

    black_pixels_percentages = getPixelsProportion(rectangles, img)
    print("Porcentagem de pixels pretos para cada retangulo envolvendo um objeto\n", black_pixels_percentages, "\n")

    transitions_proportions = getTransitionsProportion(rectangles, img)
    print("Razao das transicoes branco para preto e numero total de pixels pretos\n", transitions_proportions, "\n")

    # Deteccao de linhas
    num_linhas, text_rects = getText(black_pixels_percentages, transitions_proportions, rectangles)
    draw_text_rects(img, text_rects, title="Objetos identificados como texto")
    print(f"Ha aproximadamente {num_linhas} linhas no texto\n")

    # Deteccao de palavras
    getWords(img, img_aux)

main()