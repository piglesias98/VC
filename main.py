# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:22:29 2019

@author: Paula Iglesias Ahualli
"""

#Paquetes necesarios para la ejecución de la práctica
import cv2
import numpy as np
from matplotlib import pyplot as plt
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
    convImg=convolucion(img, kernelx, kernely, borde)
    return convImg

#Función que calcula la laplaciana de una imagen 
#Para calcular las segundas derivadas llamamos a la función derivada
def laplaciana(imagen, tam, borde=cv2.BORDER_DEFAULT):
    img = np.copy(imagen)
    imgx = derivada(img, tam, 2, 0)
    imgy = derivada(img, tam, 0, 2)
    return imgx + imgy

#Función que calcula laplaciana de gaussiana
#Llama primero a gaussiana y después a laplaciana  y normalizamos
def laplacianaGaussiana(imagen, tam, sigma, borde=cv2.BORDER_DEFAULT):
    img = np.copy(imagen)
    img1 = gaussiana(img, sigma, borde)
    img2 = laplaciana(img1, tam, borde)
    img3=  img2 *(sigma * sigma)
    cv2.normalize(img3, img3, 0, 255, cv2.NORM_MINMAX)
    return img3

#Aplica downsampling quedándose sólo con las filas y columnas impares
def submuestrear(imagen):
    down =imagen[::2, ::2]
    return down

#Crea una pirámide de Gauss aplicando el filtro gaussiano y submuestreando en cada nivel
def piramideGauss(imagen, niveles, sigma=1, borde=cv2.BORDER_DEFAULT):
    g = imagen.copy()
    gp = [imagen] #crea un array
    for i in range(niveles):
        g = gaussiana(g, sigma, borde)
        g = submuestrear(g)
        gp.append(g)
    return gp

#Función que crea una imagen que alberga todas las subimágenes de la pirámide
def imagenPiramide(imgs):
    # Filas y columnas de la imagen.
    height, width= imgs[0].shape[:2]
    # La imagen final se crea con ancho= ancho(primera) + ancho(segunda)
    # ancho(segunda) = 1/2 ancho(primera)
    img = np.zeros((height, width+math.ceil(width*0.5)))
    img[0:height, 0:width] = imgs[0]
    # Posición donde deben comenzar las imágenes
    init_col = width
    init_row = 0
    # n imágenes
    num_imgs = len(imgs)
    
    # Se recorren el resto de imágenes para colocarlas donde corresponde
    for i in range(1, num_imgs):
        # Filas y columnas de la imagen actual
        height, width= imgs[i].shape[:2]
        # Se hace el copiado de la imagen actual como se ha hecho con la primera
        img[init_row:init_row+height, init_col:init_col+width] = imgs[i]
        # Se aumenta el contador desde donde se colocará la siguiente imagen
        init_row += height
    
    return img


#Aplica upsampling duplicando las filas y columnas de la imagen
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

#Crea una pirámide laplaciana
#Para cada nivel i restamos al nivel i de la gaussiana el nivel i+1 sobremuestreado
def piramideLaplaciana(imagen, niveles, borde= cv2.BORDER_DEFAULT):
    pg = piramideGauss(imagen, niveles, borde)
    pl = [pg[niveles]]
    for i in range(niveles, 0, -1):
        gUp = sobremuestrearDuplicar(pg[i])
        height, width = pg[i-1].shape[:2]
        gUpRes = cv2.resize(gUp, (width, height))
        l = cv2.subtract(pg[i-1], gUpRes)
        #l = convolucionGaussiana(l, 3, -1)
        pl.append(l)
    pl = np.flip(pl)
    return pl

#Primero aplica laplaciana gaussiana
#normaliza
#realiza una supresion de no máximos
#llama a circulos para pintar sobre la imagen original
def busquedaRegiones(imagen, nEscalas, sigma, tam):
    img = np.copy(imagen)
    escalas = []
    for i in range(nEscalas):
        lg = laplacianaGaussiana(img, tam, sigma)
        sq = np.square(lg)
        cv2.normalize(sq, sq, 0, 255, cv2.NORM_MINMAX)
        sq = sq.astype(np.uint8)
        supresion = supresionNoMaximos(sq)
        final = circulos(img, supresion,sigma)
        escalas.append(final)
        sigma = sigma + 1
    return escalas

#para cada píxel de la imagen, miramos sus ocho vecinos
#Si alguno es mayor que el píxel referencia, lo ponemos  a 0.
def supresionNoMaximos(imagen):
    img = np.copy(imagen)
    height, width = imagen.shape[:2]
    supresion= np.zeros((height, width))
    for i in range(height -2):
       for j in range(width -2):
           #Obtenemos el máximo de los vecinos
           maximo = np.amax(img[i:(i+3), j:(j+3)])
           if maximo == img [i+1,j+1]:
               supresion[i+1,j+1] = img[i+1, j+1]
    return supresion

#en cada pixel de img_max que supere el umbral se pinta un círculo en img
def circulos(imagen, img_max ,sigma):
    umbral = 100
    img = np.copy(imagen)
    height, width = imagen.shape[:2]
    #Recorremos cada pixel
    for i in range(height):
        for j in range(width):
            #Si es mayor que el umbral, hacemos un círculo en la imagen original
            if img_max[i,j]>umbral:
                radio = int(math.sqrt(2)  * sigma)
                color = (255,0,0)
                cv2.circle(img, (j,i), radio, color)
    return img

#Calcula H con la siguiente fórmula H=I1*G1+I2*(1-G2)
def imagenesHibridas(imagen1, imagen2, lFreq, hFreq):
    img1= np.copy(imagen1)
    img2 = np.copy(imagen2)
    # Obtenemos I1 (baja frecuencia)
    i1 = gaussiana(img1, lFreq)
    # Obtenemos I2 (alta frecuencia)
    g2 = gaussiana(img2, hFreq)
    i2 =  img2 - g2
    # Hibridamos Imagen I1+I2
    h = i1+i2
    return i1, i2, h


"""
Ejercicios

"""

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
        
        derivadas.append(derivada(gato, i, 1, 0, cv2.BORDER_CONSTANT))
        tit = "sigma = "+str(i)+" Dx=1, Dy=0" + " Borde: Constant"
        titulos.append(tit)
        
        derivadas.append(derivada(gato, i, 0, 1, cv2.BORDER_CONSTANT))
        tit = "sigma = "+str(i)+" Dx=0, Dy=1" + " Borde: Constant"
        titulos.append(tit)
        
        derivadas.append(derivada(gato, i, 1, 1, cv2.BORDER_CONSTANT))
        tit = "sigma = "+str(i)+" Dx=1, Dy=1" + " Borde: Constant"
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
    
    imgs = [imagenPiramide(piramide)]
    
    piramide = piramideGauss(perro, 4, cv2.BORDER_REPLICATE)
    
    imgs.append(imagenPiramide(piramideGauss(perro, 4, cv2.BORDER_REPLICATE)))

    imgs.append(imagenPiramide(piramideGauss(perro, 4, cv2.BORDER_REFLECT)))
    
    titulos = ["Sin bordes","Borde = Replicate","Borde = Reflect"]
    
    representarImagenes(imgs, titulos, 1)
    titulos.clear()
    piramide.clear()


def ejercicio2b():
    print("Ejercicio 2 - Apartado B")
    print("Piramide laplaciana")
    print()
    
    piramide = piramideLaplaciana(perro, 4)
    
    imgs = [imagenPiramide(piramide)]
    
    piramide = piramideLaplaciana(perro, 4, cv2.BORDER_REPLICATE)
    
    imgs.append(imagenPiramide(piramideLaplaciana(perro, 4, cv2.BORDER_REPLICATE)))

    imgs.append(imagenPiramide(piramideLaplaciana(perro, 4, cv2.BORDER_REFLECT)))
    
    titulos = ["Si bordes","Borde = Replicate","Borde = Reflect"]
    
    representarImagenes(imgs, titulos, 1)
    imgs.clear()


def ejercicio2c():
    print("Ejercicio 2 - Apartado C")
    print("Búsqueda de regiones")
    print()
    escalas = busquedaRegiones(gato, 5, 5, 3)
    titulo=[]
    for i in range(5):
        titulo.append("Escala " + str(i))
    
#    res = cv2.hconcat(escalas)
    representarImagenes(escalas,titulo, 5)
    
    
def ejercicio3a():
    print("Ejercicio 3 - Apartado A")
    print("Imágenes híbridas")
    print()
    
    #Ejemplo perro gato
    i1, i2, h = imagenesHibridas(gato ,perro, 9, 5)
    titulos = ["Baja", "Alta", "Híbrida"]
    representarImagenes([i1, i2, h], titulos, 3)
    
    #Ejemplo Einstein Marilyn
    i1, i2, h = imagenesHibridas(marilyn ,einstein, 3, 3)
    titulos = ["Baja", "Alta", "Híbrida"]
    representarImagenes([i1, i2, h], titulos, 3)
    
    #Ejemplo pez submarino
    i1, i2, h = imagenesHibridas(submarino ,pez, 9, 7)
    titulos = ["Baja", "Alta", "Híbrida"]
    representarImagenes([i1, i2, h], titulos, 3)
    
    #Ejemplo pájaro avión
    i1, i2, h = imagenesHibridas(pajaro ,avion, 5, 3)
    titulos = ["Baja", "Alta", "Híbrida"]
    representarImagenes([i1, i2, h], titulos, 3)
    
    
def ejercicio3b():
    print("Ejercicio 3 - Apartado B")
    print("Pirámides híbridas")
    print()
    
    #Ejemplo pájaro avión
    i1, i2, h = imagenesHibridas(pajaro ,avion, 5, 3)
    pir = piramideGauss(h, 4, 5)
    representarImagenes([imagenPiramide(pir)], ["Piramide Pájaro-Avión"])
    #Ejemplo pájaro avión
    i1, i2, h = imagenesHibridas(marilyn ,einstein, 3, 3)
    pir = piramideGauss(h, 4, 3)
    representarImagenes([imagenPiramide(pir)], ["Piramide Marilyn-Einstein"])
    #Ejemplo pez submarino
    i1, i2, h = imagenesHibridas(submarino ,pez, 9, 7)
    pir = piramideGauss(h, 4, 7)
    representarImagenes([imagenPiramide(pir)], ["Piramide Pez-Submarino"])

    

ejercicio1a()
input("Pulse ENTER para continuar")
ejercicio1b()
input("Pulse ENTER para continuar")
ejercicio2a()
input("Pulse ENTER para continuar")
ejercicio2b()
input("Pulse ENTER para continuar")
ejercicio2c()
input("Pulse ENTER para continuar")
ejercicio3a()
input("Pulse ENTER para continuar")
ejercicio3b()