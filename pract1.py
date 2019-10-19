# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 18:44:27 2019

@author: Paula
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import math


def leeImagen(filename, flagColor):
    if flagColor == True:
        imagen = cv2.imread(filename, 1)
        #tenemos que convertir la imagen a RGB
        #imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    else:
        imagen = cv2.imread(filename, 0)
        #imagen = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        #convertimos la imagen a RGB
        #imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen


#EJERCICIO 1 A
#Usando las funciones de OpenCV 

#A) Cálculo de la convolución de una imagen con una máscara 2D
#Usar
#  - Gaussiana 2D Gaussian Blur
#  - Máscaras 1D de getDerivKernels
#Mostrar
# - Ejemplos de distintos tamaños de máscara, valores de sigma y condiciones de contorno

def ej1a(imagen, sigma, tam, dx, dy, alisamiento):
    if (alisamiento):
        #utilizamos getGaussianKernel
        imgConv =convolucionGaussiana(imagen, sigma, tam)
    else:
        #calculamos las derivadas con getDerivKernels
        imgConv = convolucionDerivada(imagen, tam, dx, dy)    
    #Realizamos la convolución con el kernel que hemos calculado
    return imgConv

def convolucionSeparada(imagen, kernel):
    # Filas y columnas de la imagen.
    height, width = imagen.shape
    convImg = imagen
    # Recorremos filas y columnas.
    convImg[0:height]=cv2.filter2D(imagen[:],-1,kernel)
    convImg[:,0:width]=cv2.filter2D(imagen[:,0:width],-1,kernel.T)

    return convImg

def convolucionGaussiana(imagen, sigma, tam):
    if tam==-1:
        tam = 6*sigma +1
    kernel = cv2.getGaussianKernel(tam, sigma)
    imgConv = convolucionSeparada(imagen, kernel)
    return imgConv

def convolucionDerivada(imagen, tam, dx, dy):
    # Filas y columnas de la imagen.
    height, width, channels = imagen.shape
    convImg = imagen
    
    #calculamos los kernels
    kernelx, kernely = cv2.getDerivKernels(dx, dy, tam)
    print(np.dot(kernelx, np.transpose(kernely)))
    #Aplicamos la convolución
    convImg[:]=cv2.filter2D(imagen[:],-1,kernelx.T)
    convImg[:,:width]=cv2.filter2D(imagen[:,:width],-1,kernely)
    return convImg


#EJERCICIO 1 B
    
def convolucionLaplaciana(imagen, tam,tipoBorde=0):
    # Obtenemos la segunda derivada en X.
    convImgX = convolucionDerivada(copy.deepcopy(imagen), tam, 2, 0,)
    # Obtenemos la segunda derivada en Y.
    convImgY = convolucionDerivada(copy.deepcopy(imagen), tam, 0, 2,)

    # Obtenemos la imagen de la laplaciana.
    convImg = convImgX + convImgY
    # Devolvemos el resultado.
    return convImg


#EJERCICIO 2 A
    
#def submuestrear(imagen):
#    rows, cols= imagen.shape
#    newRows = int(rows/2)
#    newCols = int(cols/2)
#    subm = np.zeros((newRows, newCols))
#    for i in range(newRows):
#        for j in range(newCols):
#            subm[i,j] = imagen[2*i+1, 2*j+1]
#    return subm
def submuestrear(imagen):
    down =imagen[::2, ::2]
#    print(imagen.shape)
#    print(down.shape)
    return down

def piramideGauss(imagen, niveles):
    g = imagen.copy()
    gp = [imagen] #crea un array
    for i in range(niveles):
        #img = convolucionGaussiana(piramide[i], 3, -1)
        g = submuestrear(g)
        gp.append(g)
    return gp



#EJERCICIO 2 B
def sobremuestrearCeros(imagen):
    height, width = imagen.shape
    up = np.zeros((height*2, width*2),dtype="uint8")
    up[::2, ::2]=imagen
    return up


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
    return convolucionGaussiana(dcols.T, 3, -1)

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

def pirLaplaciana(imagen, niveles):
    pg = piramideGauss(imagen, niveles)
    pl = [pg[niveles]]
    for i in range(niveles, 0, -1):
        gUp = cv2.pyrUp(pg[i])
        height, width = pg[i-1].shape[:2]
        gUpRes = cv2.resize(gUp, (width, height))
        l = cv2.subtract(pg[i-1], gUpRes)
        pl.append(l)
    return pl

#PRUEBA EJERCICIOS

rutaImagen = 'C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\marilyn.bmp'
#imagen = leeImagen(rutaImagen, False)
imagen = cv2.imread(rutaImagen, 0)
#print(imagen)
#f, c, ch = imagen.shape
#print(ch)
#plt.imshow(imagen,  cmap = plt.cm.gray)
#convolucion1 = convolucionGaussiana(imagen, 5, -1)
#convolucion1 = ej1a(imagen, -1, 3, 1, 0, False)
#convolucion1 = convolucionLaplaciana(imagen, 3)
#piramide = piramideGauss(imagen, 4)
#print(len(piramide))
#for i in piramide:
#    print(i)
#rows, cols= imagen.shape
#newRows = int(rows/2)
#newCols = int(cols/2)
#subm = np.zeros((newRows, newCols))

#plt.imshow(convolucion1)
#imagen = cv2.imread(rutaImagen)
#convolucion2 = convolucionDerivada(imagen, 3, 1, 1)
#plt.imshow(convolucion2)
 
#print(imagen)
#img = convolucionGaussiana(imagen, 3, -1)
#img2= submuestrear(img)
#print("otro")
#print(img)
#print("otro")
#print(img2)

#print(type(imagen))
#piramide = [imagen] #crea un array
#for i in range(3):
#    print(i)
#    print("conv gauss")
#    img = convolucionGaussiana(piramide[i], 3, -1)
#    print(type(img))
#    print(img)
#    print("subm")
#    img2 = submuestrear(img)
#    print(type(img2))
#    piramide.append(img)
  

#plt.imshow(piramide[0])
#plt.imshow(piramide[1])
#plt.imshow(piramide[2])
#plt.imshow(piramide[3])
#plt.imshow(piramide[4])
#plt.show()



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
    
    

def display_nuevo(img):
    for im in img:
        imgt = (np.clip(im,0,1)*255.).astype(np.uint8)
        height, width = imgt.shape
        figsize = width, height
        plt.figure(figsize=figsize)
        plt.imshow(imgt)
        plt .close()
    plt.show()
    
    
#display_piramide(piramide)
#for i in range(len(piramide)):
#    print(i)
#    #print(piramide[i])
#    img = piramide[i]
#    print(type(img))
#    plt.imshow(img)
#    cv2.waitKey()
def representar_imagenes(lista_imagen_leida, lista_titulos):

	# Comprobamos que el numero de imágenes corresponde con el número de títulos pasados
	if len(lista_imagen_leida) != len(lista_titulos):
		print("No hay el mismo número de imágenes que de títulos.")
		return -1 # No hay el mismo numero de imágenes que de títulos

	# Calculamos el numero de imagenes
	n_imagenes = len(lista_imagen_leida)

	# Establecemos por defecto el numero de columnas
	n_columnas = 1

	# Calculamos el número de filas
	n_filas = (n_imagenes // n_columnas) + (n_imagenes % n_columnas)

	# Establecemos por defecto un tamaño a las imágenes
	plt.figure(figsize=(100,100))

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
    
    
#
piramide = piramideLaplaciana(imagen, 10)
print("AHORA ESTAMOS EN PIRAMIDE")
titulos = ["1","2","3","4","5","6","7","8","9","10", "11"]
representar_imagenes(piramide, titulos)

#
#x = np.array([[0,1,2,3,4],
#              [5,6,7,8,9],
#              [10,11,12,13,14],
#              [15,16,17,18,19]])

#x = np.array([[1,2],[3,4]])
#y = sobremuestrear(x)
#print(y)
#z = cv2.pyrUp(x)
#print(z)
#niveles=10
#pg = piramideGauss(imagen, niveles)
#pl = [pg[niveles]]
#for i in range(niveles, 0, -1):
##    gUp = cv2.pyrUp(pg[i])
#    gUp = sobremuestrearDuplicar(pg[i])
#    plt.imshow(gUp, cmap="gray")
#    height, width = pg[i-1].shape[:2]
#    gUpRes = cv2.resize(gUp, (width, height))
#    l = cv2.subtract(pg[i-1], gUpRes)
#    #l = convolucionGaussiana(l, 3, -1)
#    pl.append(l)
#    
#plt.imshow(pl[10], cmap="gray")
#    
    
kernel = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[17,18,19,20,21],[22,23,25,26,27]])   
#print(kernel)
#height, width = kernel.shape
#print("concantemanos")
##up =np.concatenate(([kernel[0]], [kernel[0]]), axis=0)
##fila =np.concatenate(([kernel[1]], [kernel[1]]), axis=0)
##up= np.append(up, fila, axis=0)
##np.append(up, np.concatenate(([kernel[1]], [kernel[1]]), axis=0))
#up =np.concatenate(([kernel[0]], [kernel[0]]), axis=0)
#for i in range(1, height):
#    fila =np.concatenate(([kernel[i]], [kernel[i]]), axis=0)
#    up= np.append(up, fila, axis=0)
#    
#print(up)
#print("columnas")
##ap = np.concatenate(([kernel[:,0]], [kernel[:,0]]), axis=0)
##columna = np.concatenate(([kernel[:,1]], [kernel[:,1]]), axis=0)
##ap= np.append(ap, columna, axis=0)
##ap = ap.T
#print(kernel)
#
#ap = np.concatenate(([up[:,0]], [up[:,0]]), axis=0)
#for j in range(1, width):
#    columna = np.concatenate(([up[:,j]], [up[:,j]]), axis=0)
#    ap= np.append(ap, columna, axis=0)
#    
#ap=ap.T
#print(ap)
#print(sobremuestrearDuplicar(kernel))