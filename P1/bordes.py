'''
	4. Detectores Borde - Esquina
'''
from math import *
import numpy as np
from progress import progress
from filtrado import convolve, gaussianFilter2D

'''
	4. Bordes y Esquinas
'''

def robertsKern():
	gx = np.matrix([[-1, 0], [0, 1]])
	gy = np.matrix([[0, -1], [1, 0]])
	return gx, gy

def centralDiffKern():
	gx = np.matrix([-1, 0, 1])
	gy = gx.T
	return gx, gy

def prewittKern():
	gx = np.matrix([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
	gy = gx.T
	return gx, gy

def sobelKern():
	gx = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	gy = gx.T
	return gx, gy

def derivatives(inputImage, operator):
	'''
		Implementar una función que permita obtener las componentes de gradiente Gx
		y Gy de una imagen, pudiendo elegir entre los operadores de Roberts,
		CentralDiff (Diferencias centrales de Prewitt/Sobel sin promedio), Prewitt y Sobel.
	'''

	gx, gy = None, None

	if operator == 'Roberts':
		kx, ky = robertsKern()
		print(">> Obteniendo gradiente en x")
		gx = convolve(inputImage, kx)
		print(">> Obteniendo gradiente en y")
		gy = convolve(inputImage, ky)
	elif operator == 'CentralDiff':
		kx, ky = centralDiffKern()
		print(">> Obteniendo gradiente en x")
		gx = convolve(inputImage, kx)
		print(">> Obteniendo gradiente en y")
		gy = convolve(inputImage, ky)
	elif operator == 'Prewitt':
		kx, ky = prewittKern()
		print(">> Obteniendo gradiente en x")
		gx = convolve(inputImage, kx)
		print(">> Obteniendo gradiente en y")
		gy = convolve(inputImage, ky)
	elif operator == 'Sobel':
		kx, ky = sobelKern()
		print(">> Obteniendo gradiente en x")
		gx = convolve(inputImage, kx)
		print(">> Obteniendo gradiente en y")
		gy = convolve(inputImage, ky)
	else:
		print('> Método no definido')

	return [gx, gy]

def edgeCanny(inputImage , sigma , tlow , thigh):
	'''
		Implementar el detector de bordes de Canny mediante una función que
		permita especificar el valor de σ en el suavizado Gaussiano y los
		umbrales del proceso de histéresis
	'''

	# Obtengo las dimensiones
	width, height = inputImage.shape[0], inputImage.shape[1]

	# 1. Suavizado
	print('> 1.1 suavizado')
	smooth = gaussianFilter2D(inputImage, sigma)
	#smooth = inputImage

	# 2. Detección de bordes
	print('> 1.2 Detección de bordes')
	[gx, gy] = derivatives(smooth, 'Sobel')
	# Se calcula la magnitud
	print('> 1.2.1 Cálculo de la magnitud')
	magnitude = np.sqrt((gx ** 2) + (gy ** 2))
	# Se calcula la orientación
	print('> 1.2.2 Cálculo de la orientación')
	orientation = np.arctan2(gy, gx)
	print('> 1.2.3 Discretización de ángulos')
	for i in range(0, width):
		for j in range(0, height):
			# Feedback
			progress(i*height+j, width*height)

			# Obtiene el ángulo
			value = abs(orientation[i, j])

			# Discretiza el ángulo
			if value >= pi/8 and value < 3*pi/8:
				orientation[i, j] = 45
			elif value >= 3*pi/8 and value < 5*pi/8:
				orientation[i, j] = 90
			elif value >= 5*pi/8 and value < 7*pi/8:
				orientation[i, j] = 135
			else:
				orientation[i, j] = 0
	print()

	# 3. Supresión no máxima

	# 4. Histéresis

	return magnitude, orientation
