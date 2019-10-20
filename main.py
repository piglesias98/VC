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

#Leer imágenes
def leerImagenes():
    bicicleta = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\bicyle.bmp', 0)
    pajaro = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\bird.bmp', 0)
    gato = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\cat.bmp', 0)
    perro = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\dog.bmp', 0)
    einstein = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\einstein.bmp', 0)
    pez = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\fish.bmp', 0)
    marilyn = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\marilyn.bmp', 0)
    moto = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\motorcycle.bmp', 0)
    avion = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\plane.bmp', 0)
    submarino = cv2.imread('C:\\Users\\Paula\\Documents\\VC\\practica1\\images\\submarine.bmp', 0)


#EJERCICIO 1 A
def ejercicio1a():
    