# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:22:29 2019

@author: Paula
"""

#Paquetes necesarios para la ejecución de la práctica
import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import math

"""
Configuraciones previas

"""

cmap = cv2.COLOR_RGB2GRAY
path = "C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\"

#Función que asigna un esquema de colores. Por defecto escala de grises
def set_c_map(img, cmap = cv2.COLOR_RGB2GRAY):
    img = cv2.cvtColor(img, cmap)
    return img

#Leemos las imágenes que vamos a usar
    
gato = cv2.imread(path+'cat.bmp')
gato = set_c_map(gato, cmap)
gato = gato.astype(float)

perro = cv2.imread(path+'dog.bmp')
perro = set_c_map(perro, cmap)
perro = perro.astype(float)

avion = cv2.imread(path+'plane.bmp')
avion = set_c_map(avion, cmap)
avion = avion.astype(float)

pajaro = cv2.imread(path+'bird.bmp')
pajaro = set_c_map(pajaro, cmap)
pajaro = pajaro.astype(float)

einstein = cv2.imread(path+'einstein.bmp')
einstein = set_c_map(einstein, cmap)
einstein = einstein.astype(float)

marilyn = cv2.imread(path+'marilyn.bmp')
marilyn = set_c_map(marilyn, cmap)
marilyn = marilyn.astype(float)

pez = cv2.imread(path+'fish.bmp')
pez = set_c_map(pez, cmap)
pez = pez.astype(float)

moto = cv2.imread(path+'motorcycle.bmp')
moto = set_c_map(moto, cmap)
moto = moto.astype(float)

submarino = cv2.imread(path+'submarine.bmp')
submarino = set_c_map(submarino, cmap)
submarino = submarino.astype(float)

bicicleta = cv2.imread(path+'bicycle.bmp')
bicicleta = set_c_map(bicicleta, cmap)
bicicleta = bicicleta.astype(float)


#Función que representa las imágenes de un array en las columnas y el tamaño que se le indique
def representarImagenes(lista_imagen_leida, lista_titulos, n_col=2, tam=15):

	# Comprobamos que el numero de imágenes corresponde con el número de títulos pasados
	if len(lista_imagen_leida) != len(lista_titulos):
		print("No hay el mismo número de imágenes que de títulos.")
		return -1 # No hay el mismo numero de imágenes que de títulos

	# Calculamos el numero de imagenes
	n_imagenes = len(lista_imagen_leida)

	# Calculamos el número de filas
	n_filas = (n_imagenes // n_col) + (n_imagenes % n_col)

	# Establecemos por defecto un tamaño a las imágenes
	plt.figure(figsize=(tam,tam))

	# Recorremos la lista de imágenes
	for i in range(0, n_imagenes):

		plt.subplot(n_filas, n_col, i+1) # plt.subplot empieza en 1

		if (len(np.shape(lista_imagen_leida[i]))) == 2: # Si la imagen es en gris
			plt.imshow(lista_imagen_leida[i], cmap = 'gray')
		else: # Si la imagen es en color
			plt.imshow(cv2.cvtColor(lista_imagen_leida[i], cv2.COLOR_BGR2RGB))

		plt.title(lista_titulos[i]) # Añadimos el título a cada imagen

		plt.xticks([]), plt.yticks([]) # Para ocultar los valores de tick en los ejes X e Y

	plt.show()

"""
Funciones para los ejercicios

"""
#Realiza la convolución 2D de un kernel x y kernel y, tanto por filas como por columnas    
def convolucion(imagen, kernelx, kernely, border=cv2.BORDER_DEFAULT, normalize=True):
    kernelx = np.flip(kernelx)
    kernely= np.flip(kernely)
    img = np.copy(imagen)
    height, width = imagen.shape[:]
    for i in range(height):
        resConv = cv2.filter2D(img[i, :], -1, kernelx, border)
        img[i, :] = [sublist[0] for sublist in resConv]
    for j in range(width):
        resConv = cv2.filter2D(img[:, j], -1, kernely, border)
        img[:, j] = [sublist[0] for sublist in resConv]
    if normalize:
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    return img

#Función que aplica un filtro gaussiano de un sigma dado a la imagen
def gaussiana(imagen, sigma, borde=cv2.BORDER_DEFAULT):
    img = np.copy(imagen)
    tam = 6*sigma +1
    kernel = cv2.getGaussianKernel(tam, sigma)
    imgConv = convolucion(img, kernel, kernel, borde)
    return imgConv

#Función que realiza la convolución de una imagen con los kernels de la derivada del orden que se indique
def derivada(imagen, tam, dx, dy, borde=cv2.BORDER_DEFAULT):
    # Filas y columnas de la imagen.
    height, width= imagen.shape[:2]
    img = np.copy(imagen)
    #calculamos los kernels
    kernelx, kernely = cv2.getDerivKernels(dx,dy,tam, borde)
    #Realizamos la convolucion
    convImg=convolucion(img, kernelx, kernely)
    return convImg

def laplaciana(imagen, tam, borde=cv2.BORDER_DEFAULT):
    img = np.copy(imagen)
    imgx = derivada(img, tam, 2, 0)
    imgy = derivada(img, tam, 0, 2)
    return imgx + imgy

def laplacianaGaussiana(imagen, tam, sigma, borde=cv2.BORDER_DEFAULT):
    img = np.copy(imagen)
    img1 = gaussiana(img, sigma, borde)
    img2 = laplaciana(img1, tam, borde)
    img3=  img2 *(sigma * sigma)
    cv2.normalize(img3, img3, 0, 255, cv2.NORM_MINMAX)
    return img3


def submuestrear(imagen):
    down =imagen[::2, ::2]
    return down

def piramideGauss(imagen, niveles, borde=cv2.BORDER_DEFAULT):
    g = imagen.copy()
    gp = [imagen] #crea un array
    for i in range(niveles):
        g = gaussiana(g, 1, borde)
        g = submuestrear(g)
        gp.append(g)
    return gp

def show_pyr(imgs):
    """

    Función que muestra una serie de imágenes que recibe en una lista por
    parámetro en forma de pirámide.
    Como requisito para su correcto funcionamiento, las imágenes deben
    decrementar su tamaño en la mitad a medida que ocupan una posición posterior
    en la lista.

    Devuelve una sola imagen con forma de pirámide donde se encuentran todas las
    recibidas.

    """

    # Se crea una imagen inicialmente vacía que albergará todas las subimágenes
    # que se reciben.
    # El ancho de la imagen general será el ancho de la primera más el ancho
    # de la segunda (que será la mitad de la primera).

    # El ancho se calcula como len(img[0])+len(img[0])*0.5
    shape = imgs[0].shape

    height = shape[0]
    width = shape[1]

    # Se crea la imagen general con las medidas para que entren todas
    img = np.zeros((height, width+math.ceil(width*0.5)))

    # Se copia la primera imagen desde el punto de partida hasta el tamaño que
    # tiene
    img[0:height, 0:width] = imgs[0]

    # Se guarda la posición desde donde deben comenzar las imágenes
    init_col = width
    init_row = 0

    # Número de imágenes
    num_imgs = len(imgs)

    # Se recorren el resto de imágenes para colocarlas donde corresponde
    for i in range(1, num_imgs):

        # Se consigue el tamaño de la imagen actual
        shape = imgs[i].shape

        height = shape[0]
        width = shape[1]

        # Se hace el copiado de la imagen actual como se ha hecho con la primera
        img[init_row:init_row+height, init_col:init_col+width] = imgs[i]

        # Se aumenta el contador desde donde se colocará la siguiente imagen
        init_row += height

    return img

def display_piramide(img,color=True):
    
    for im in img:
        imgt = (np.clip(im,0,1)*255.).astype(np.uint8)
        if color:
            nimg = cv2.cvtColor(imgt, cv2.COLOR_BGR2RGB)
        else:
            nimg = cv2.cvtColor(imgt,cv2.COLOR_GRAY2RGB)
        dpi = 50
        height, width, depth = nimg.shape
    
        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)
    
        # Create a figure of the right size with one axes that takes up the full figure
        plt.figure(figsize=figsize)
        #ax = fig.add_axes([0, 0, 1, 1])
    
        # Hide spines, ticks, etc.
        #ax.axis('off')
    
        # Display the image.
        plt.imshow(nimg)
    plt.show()


def sobremuestrearDuplicar(imagen):
    height, width = imagen.shape
    
    #duplicamos las filas
    drows =np.concatenate(([imagen[0]], [imagen[0]]), axis=0)
    for i in range(1, height):
        row =np.concatenate(([imagen[i]], [imagen[i]]), axis=0)
        drows= np.append(drows, row, axis=0)
        
    #duplicamos las columnas
    dcols = np.concatenate(([drows[:,0]], [drows[:,0]]), axis=0)
    for j in range(1, width):
        col = np.concatenate(([drows[:,j]], [drows[:,j]]), axis=0)
        dcols= np.append(dcols, col, axis=0)
        
    return gaussiana(dcols.T, 3, -1)

def piramideLaplaciana(imagen, niveles):
    pg = piramideGauss(imagen, niveles)
    pl = [pg[niveles]]
    for i in range(niveles, 0, -1):
        gUp = sobremuestrearDuplicar(pg[i])
        height, width = pg[i-1].shape[:2]
        gUpRes = cv2.resize(gUp, (width, height))
        l = cv2.subtract(pg[i-1], gUpRes)
        #l = convolucionGaussiana(l, 3, -1)
        pl.append(l)
    return pl



#EJERCICIO 1 Aol
def ejercicio1a():
    print("Ejercicio 1 - Apartado A")
    print("Máscara gaussiana")
    print()
    
    sigma3 = gaussiana(pajaro, 3)
    imagenes = [sigma3]
    titulos = ["Sigma = 3, Borde = default"]
    
    sigma3 = gaussiana(pajaro, 3, cv2.BORDER_CONSTANT)
    imagenes.append(sigma3)
    titulos.append("Sigma = 3, Borde = Replicate")
    
    sigma5= gaussiana(pajaro, 5)
    imagenes.append(sigma5)
    titulos.append("Sigma = 5")
    
    sigma5= gaussiana(pajaro, 5, cv2.BORDER_CONSTANT)
    imagenes.append(sigma5)
    titulos.append("Sigma = 5, Borde = Replicate")
    
    representarImagenes(imagenes, titulos, 2)
    imagenes.clear()
    titulos.clear()
    
    
    print("Máscara derivada")
    print()
    
    derivadas =[]
    titulos = []
    for i in range(3, 6, 2):
        derivadas.append(derivada(gato, i, 1, 0))
        tit = "sigma = "+str(i)+" Dx=1, Dy=0" 
        titulos.append(tit)
        
        derivadas.append(derivada(gato, i, 0, 1))
        tit = "sigma = "+str(i)+" Dx=0, Dy=1" 
        titulos.append(tit)
        
        derivadas.append(derivada(gato, i, 1, 1))
        tit = "sigma = "+str(i)+" Dx=1, Dy=1" 
        titulos.append(tit)
        
    representarImagenes(derivadas, titulos, 3)
    derivadas.clear()
    

def ejercicio1b():
    print("Ejercicio 1 - Apartado B")
    print("Laplaciana de gaussiana")
    print()
    
    imagenes=[laplacianaGaussiana(submarino, 5, 1, cv2.BORDER_REPLICATE)]
    titulos=["Sigma= 1, Borde=Replicate"]
    
    imagenes.append(laplacianaGaussiana(submarino, 5, 3, cv2.BORDER_REPLICATE))
    titulos.append("sigma = 3, Borde=Replicate")
    
    imagenes.append(laplacianaGaussiana(submarino, 5, 1, cv2.BORDER_CONSTANT))
    titulos.append("sigma = 1, Borde= Constant")
    
    imagenes.append(laplacianaGaussiana(submarino, 5, 3, cv2.BORDER_CONSTANT))
    titulos.append("sigma = 3, Borde= Constant")
    
    representarImagenes(imagenes, titulos, 2)
    imagenes.clear()
    titulos.clear()
    
    
def ejercicio2a():
    print("Ejercicio 2 - Apartado A")
    print("Piramide de gaussiana")
    print()
    
    piramide = piramideGauss(perro, 4)
    
    imgs = [show_pyr(piramide)]
    
    piramide = piramideGauss(perro, 4, cv2.BORDER_REPLICATE)
    
    imgs.append(show_pyr(piramideGauss(perro, 4, cv2.BORDER_REPLICATE)))

    imgs.append(show_pyr(piramideGauss(perro, 4, cv2.BORDER_REFLECT)))
    
    titulos = ["Si bordes","Borde = Replicate","Borde = Reflect"]
    
    representarImagenes(imgs, titulos, 1)


def ejercicio2b():
    print("Ejercicio 2 - Apartado B")
    print("Piramide laplaciana")
    print()
    
    piramide = piramideLaplaciana(pez, 4)
    
#    display_piramide(piramide, False)
    titulos = ["Si bordes","Borde = Replicate","Borde = Reflect", "a","a"]
    representarImagenes(piramide, titulos, 1)
#    piramide = piramideLaplaciana(perro, 4, cv2.BORDER_REPLICATE)
    
#    imgs.append(show_pyr(piramideLaplaciana(pez, 4, cv2.BORDER_REPLICATE)))
#
#    imgs.append(show_pyr(piramideLaplaciana(pez, 4, cv2.BORDER_REFLECT)))
#    
#    titulos = ["Si bordes","Borde = Replicate","Borde = Reflect"]
#    
#    representarImagenes(imgs, titulos, 1)
#    
    
#ejercicio1a()
#input("Pulse ENTER para continuar")
#ejercicio1b()
#input("Pulse ENTER para continuar")
#ejercicio2a()
#input("Pulse ENTER para continuar")
ejercicio2b()
