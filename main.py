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

def representarImagenes(lista_imagen_leida, lista_titulos, n_col):

	# Comprobamos que el numero de imágenes corresponde con el número de títulos pasados
	if len(lista_imagen_leida) != len(lista_titulos):
		print("No hay el mismo número de imágenes que de títulos.")
		return -1 # No hay el mismo numero de imágenes que de títulos

	# Calculamos el numero de imagenes
	n_imagenes = len(lista_imagen_leida)

	# Establecemos por defecto el numero de columnas
	n_columnas = n_col

	# Calculamos el número de filas
	n_filas = (n_imagenes // n_columnas) + (n_imagenes % n_columnas)

	# Establecemos por defecto un tamaño a las imágenes
	plt.figure(figsize=(15,15))

	# Recorremos la lista de imágenes
	for i in range(0, n_imagenes):

		plt.subplot(n_filas, n_columnas, i+1) # plt.subplot empieza en 1

		if (len(np.shape(lista_imagen_leida[i]))) == 2: # Si la imagen es en gris
			plt.imshow(lista_imagen_leida[i], cmap = 'gray')
		else: # Si la imagen es en color
			plt.imshow(cv2.cvtColor(lista_imagen_leida[i], cv2.COLOR_BGR2RGB))

		plt.title(lista_titulos[i]) # Añadimos el título a cada imagen

		plt.xticks([]), plt.yticks([]) # Para ocultar los valores de tick en los ejes X e Y

	plt.show()

def convolucionGaussiana(imagen, sigma):
    img = np.copy(imagen)
    tam = 6*sigma +1
    kernel = cv2.getGaussianKernel(tam, sigma)
    imgConv = convolucionSeparada(img, kernel)
    return imgConv

def convolucionSeparada(imagen, kernel):
    img = np.copy(imagen)
    # Filas y columnas de la imagen.
    height, width = img.shape[:2]
    convImg = img
    # Recorremos filas y columnas.
    convImg[0:height]=cv2.filter2D(img[:],-1,kernel)
    convImg[:,0:width]=cv2.filter2D(img[:,0:width],-1,kernel.T)

    return convImg

def convolucionDerivada(imagen, tam, dx, dy):
    img = np.copy(imagen)
    # Filas y columnas de la imagen.
    height, width= img.shape
    convImg = img
    
    #calculamos los kernels
    kernelx, kernely = cv2.getDerivKernels(dx, dy, tam)
    #print(np.dot(kernelx, np.transpose(kernely)))
    #Aplicamos la convolución
    convImg[:]=cv2.filter2D(img[:],-1,kernelx.T)
    convImg[:,:width]=cv2.filter2D(img[:,:width],-1,kernely)
    return convImg


#Leer imágenes


pajaro = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\bird.bmp', 0)
gato = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\cat.bmp', 0)
perro = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\dog.bmp', 0)
einstein = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\einstein.bmp', 0)
pez = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\fish.bmp', 0)
marilyn = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\marilyn.bmp', 0)
moto = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\motorcycle.bmp', 0)
avion = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\plane.bmp', 0)
submarino = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\submarine.bmp', 0)
bicicleta = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\bicyle.bmp', 0)


#EJERCICIO 1 Aol
def ejercicio1a():
    print("Ejercicio 1 - Apartado A")
    print("Máscara gaussiana")
    sigma3 = convolucionGaussiana(pajaro, 3)
    imagenes = [sigma3]
    titulos = ["Sigma = 3"]
    sigma5= convolucionGaussiana(pajaro, 5)
    imagenes.append(sigma5)
    titulos.append("Sigma = 5")
    representarImagenes(imagenes, titulos, 2)
    print("Máscara derivada")
    derivadas =[]
    titulos = []
    for i in range(3, 6, 2):
        derivadas.append(convolucionDerivada(gato, i, 1, 0))
        tit = "sigma = "+str(i)+" Dx=1, Dy=0"
        titulos.append(tit)
        derivadas.append(convolucionDerivada(gato, i, 0, 1))
        tit = "sigma = "+str(i)+" Dx=0, Dy=1"
        titulos.append(tit)
        derivadas.append(convolucionDerivada(gato, i, 1, 1))
        tit = "sigma = "+str(i)+" Dx=1, Dy=1"
        titulos.append(tit)
    representarImagenes(derivadas, titulos, 3)

ejercicio1a()
        