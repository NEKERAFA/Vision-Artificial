'''
	1. Funciones para modificar el histograma
'''

import numpy as np

def histEnhance( inputImage, cenValue, winSize ):
	'''
		Algoritmo realce de contraste "window-level contrast enhancement",
		especificando el nivel de gris central y el tamaño de ventana.
	'''

	# Obtengo las dimensiones
	width, height = inputImage.shape[0], inputImage.shape[1]
	# Creo la imagen de salida
	outputImage = np.zeros([width, height])
	# Obtengo los valores
	minOutput = cenValue - winSize/2
	maxOutput = cenValue + winSize/2
	# Obtengo la pendiente y el valor b de la recta
	m = 255 / (maxOutput - minOutput)
	b = -m * minOutput

	# Recorro los píxeles de la imagen
	for i in range(0, width):
		for j in range(0, height):
			# Obtengo el valor de entrada
			inputValue = inputImage[i, j]
			# Transformo el valor al de salida
			outputValue = m * inputValue + b
			# Capo para que no sobrepase el mínimo y el máximo valor de blanco y negro
			outputImage[i, j] = int(max(min(outputValue, 255), 0))

	# Devuelvo la imagen
	return outputImage

def histAdapt( inputImage, minValue, maxValue ):
	'''
		Algoritmo de compresión/estiramiento de histograma, que permita
		introducir los nuevos lı́mites inferior y superior.
	'''

	# Obtengo las dimensiones
	width, height = inputImage.shape[0], inputImage.shape[1]
	# Creo la imagen de salida
	outputImage = np.zeros([width, height])
	# Obtengo los valores máximo y mínimo del diagrama
	minInput = inputImage.min()
	maxInput = inputImage.max()

	# Recorro los píxeles de la imagen
	for i in range(0, width):
		for j in range(0, height):
			# Obtengo el valor de entrada
			inputValue = inputImage[i, j]
			# Transformo el valor al de salida
			outputValue = minValue + (maxValue-minValue)*(inputValue-minInput)/(maxInput-minInput)
			# Capo para que no sobrepase el mínimo y el máximo valor de blanco y negro
			outputImage[i, j] = int(outputValue)

	# Devuelvo la imagen
	return outputImage
