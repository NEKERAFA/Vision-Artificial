'''
	1. Funciones para modificar el histograma
'''

import numpy as np
from progress import *
from timing import *

@timing
def histEnhance( inputImage, cenValue, winSize ):
	'''
		Algoritmo realce de contraste "window-level contrast enhancement",
		especificando el nivel de gris central y el tamaño de ventana.
	'''

	# Obtengo las dimensiones
	filas, columnas = inputImage.shape
	# Creo la imagen de salida
	outputImage = np.zeros(inputImage.shape)
	# Obtengo los valores
	minOutput = cenValue - winSize/2
	maxOutput = cenValue + winSize/2
	# Obtengo la pendiente y el valor b de la recta
	m = 255 / (maxOutput - minOutput)
	b = -m * minOutput

	# Recorro los píxeles de la imagen
	for i in range(0, filas):
		for j in range(0, columnas):
			# Feedback
			progress(i*filas+j, filas*columnas)
			# Obtengo el valor de entrada
			inputValue = inputImage[i, j]
			# Transformo el valor al de salida
			outputValue = m * inputValue + b
			# Capo para que no sobrepase el mínimo y el máximo valor de blanco y negro
			outputImage[i, j] = int(max(min(outputValue, 255), 0))
	print()
	# Devuelvo la imagen
	return outputImage

@timing
def histAdapt( inputImage, minValue, maxValue ):
	'''
		Algoritmo de compresión/estiramiento de histograma, que permita
		introducir los nuevos lı́mites inferior y superior.
	'''

	# Obtengo las dimensiones
	filas, columnas = inputImage.shape
	# Creo la imagen de salida
	outputImage = np.zeros(inputImage.shape)
	# Obtengo los valores máximo y mínimo del diagrama
	minInput = inputImage.min()
	maxInput = inputImage.max()

	# Recorro los píxeles de la imagen
	for i in range(0, filas):
		for j in range(0, columnas):
			# Feedback
			progress(i*filas+j, filas*columnas)
			# Obtengo el valor de entrada
			inputValue = inputImage[i, j]
			# Transformo el valor al de salida
			outputValue = minValue + (maxValue-minValue)*(inputValue-minInput)/(maxInput-minInput)
			# Capo para que no sobrepase el mínimo y el máximo valor de blanco y negro
			outputImage[i, j] = int(outputValue)
	print()
	# Devuelvo la imagen
	return outputImage
