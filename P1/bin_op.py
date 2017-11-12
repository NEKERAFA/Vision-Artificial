'''
	3. Funciones de operadores morfológicos
'''

import numpy as np
from filters import convolve
from progress import *
from timing import *

'''
	Implementar los operadores morfológicos de erosión, dilatación, apertura y
	cierre para imágenes binarias. Ambas funciones deben permitir especificar el
	tamaño del elemento estructurante y su forma (cuadrada, cruz, línea
	horizontal o vertical).
	donde ElType = 'square' | 'cross' | 'linev' | 'lineh'
'''

def EE( ElType, size ):
	'''
		Creo el elemento estucturante según el formato
	'''

	if ElType == 'square':
		return np.ones([size, size])
	elif ElType == 'linev':
		return np.ones([size, 1])
	elif ElType == 'lineh':
		return np.ones([1, size])
	elif ElType == 'cross':
		kernel = np.zeros([size, size])
		kernel[size//2] = np.ones(size)
		kernel[0:size, size//2] = np.ones([1, size])
		'''
		for i in range(0, size):
			kernel[i, size // 2] = 1
		'''
		return kernel

def dilate( inputImage, ElType, size ):
	# Creo el elmento estructurante
	kernel = EE(ElType, size)
	# Obtengo las dimensiones
	filas, columnas = inputImage.shape
	# Convoluciono
	outputImage = convolve(inputImage, kernel)

	# Aplico un umbral en 1 para dejar una imagen binaria
	for i in range(0, filas):
		for j in range(0, columnas):
			# Feedback
			progress(i*filas+j, filas*columnas)
			outputImage[i][j] = max(min(outputImage[i][j], 1), 0)
	print()
	return outputImage

def erode( inputImage, ElType, size ):
	# Creo el elmento estructurante
	kernel = EE(ElType, size)
	# Obtengo el número de elementos a 1 del kernel
	size = kernel.sum()
	# Obtengo las dimensiones
	filas, columnas = inputImage.shape
	# Convoluciono
	outputImage = convolve(inputImage, kernel)

	# Miro si la cantidad de elementos a 1 en la imagen de salida es igual al
	# del EE para dejar la imagen binaria
	for i in range(0, filas):
		for j in range(0, columnas):
			# Feedback
			progress(i*filas+j, filas*columnas, columnas)
			outputImage[i][j] = 1 if outputImage[i][j] == size else 0
	print()
	return outputImage

def opening( inputImage, ElType, size ):
	print(">> Procesando erosión")
	outputImage = erode(inputImage, ElType, size)
	print(">> Procesando dilatación")
	outputImage = dilate(outputImage, ElType, size)
	return outputImage

def closing( inputImage, ElType, size ):
	print(">> Procesando dilatación")
	outputImage = dilate(inputImage, ElType, size)
	print(">> Procesando erosión")
	outputImage = erode(outputImage, ElType, size)
	return outputImage

@timing
def tophatFilter( inputImage, ElType, size, mode ):
	'''
		Algoritmo basado en operadores morfológicos que permite calcular las
		transformaciones Top Hat blanca y negra.
		El filtro Top Hat blanco se define como la diferencia entre la imagen
		original y una apertura mientras que el negro es la diferencia entre un
		cierre y la imagen original.
	'''

	outputImage = None

	if mode == 'white':
		outputImage = inputImage - opening(inputImage, ElType, size)
	elif mode == 'black':
		outputImage = closing(inputImage, ElType, size) - inputImage
	else:
		print(">> Método no reconocido " + mode)

	return outputImage
