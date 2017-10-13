'''
	Archivo con funciones de ejemplo
'''

import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt

import histograma

original = misc.imread('example1.png', True)

def originalHist():
	# Muestro el histograma de la imagen original
	plt.figure(1)
	plt.hist(original.ravel(), range(0, 256), label='Histograma original')

def outputHist(output):
	# Muestro el histograma de la imagen original
	plt.figure(2)
	plt.hist(output.ravel(), range(0, 256), label='Histograma de salida')

	# Muestro la imagen final
	plt.figure(3)
	plt.imshow(output, cmap='gray', vmin = 0, vmax = 255)
	plt.show()

def histEnhanceExample():
	originalHist()
	# Obtengo solo los valores del medio de la imagen
	output = histograma.histEnhance(original, 128, 128)
	outputHist(output)

def histAdaptExample():
	originalHist()
	# Obtengo solo los valores del medio de la imagen
	output = histograma.histAdapt(original, 96, 160)
	outputHist(output)
