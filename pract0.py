# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:53:43 2019

@author: Paula
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


prueba1= cv2.getDerivKernels(1,0,3)
print(prueba1)
prueba1[1]

vector = [1,2,3,4,5,6,7]
matriz = np.zeros((7,3))

print(vector)
print(matriz)

primer_numero=0
for i in range(matriz.shape[0]):
    l = primer_numero
    for j in range(matriz.shape[1]):
        matriz[i][j]=vector[l]
        if l<matriz.shape[1]:
            l = l + 1
    primer_numero = primer_numero + 1  

print(vector)
print(matriz)