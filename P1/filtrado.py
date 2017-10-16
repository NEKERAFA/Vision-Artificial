'''
	2. Funciones de suavizado y realce
'''

import math
import numpy as np

def convolve( inputImage, kernel ):
	'''
		Permite realizar un filtrado espacial sobre una imagen con un kernel
		arbitrario que se pasará por parámetro.
	'''

	# Obtengo las dimensiones
	width = len(inputImage), len(inputImage[0])
	width_ker, height_ker = len(kernel), len(kernel[0])
	# Creo la imagen de salida
	outputImage = np.zeros([width, height])

	# Recorro los píxeles de la imagen
	for i in range(0, width):
		for j in range(0, height):
			'''
			x_min = max(i - (width_ker // 2), 0)
			x_max = min(i + (width_ker // 2), width)
			y_min = max(j - (width_ker // 2), 0)
			y_max = min(j + (width_ker // 2), width)

			computeImput = inputImage[x_min:x_max, y_min:y_max]
			computeKernel = kernel[]

			outputImage[i][j] = np.sum(np.multiply(inputImage[x_min:x_max, y_min:y_max], kernel[]))

			'''
			value = 0

			for k in range(0, width_ker):
				for l in range(0, height_ker):
					x = i - (width_ker // 2) + k
					y = j - (height_ker // 2) + l

					if x >= 0 and x < width and y >= 0 and y < height:
						value = value + inputImage[x][y] * kernel[k][l]

			outputImage[i][j] = value

	return outputImage

def gaussKernel1D(sigma):
	'''
		calcule un kernel Gaussiano horizontal 1 × N , a partir de σ que será
		pasado como parámetro, y calculando N como N = 2 ceil|3σ| + 1.
	'''

	# Obtengo la dimensión
	n = 2 * math.ceil(3 * sigma) + 1
	# Creo el kernel de salida
	kernel = np.zeros((n,))

	for i in range(0, n):
		x = i - (n // 2)
		kernel[i] = math.exp(-x**2 / (2 * (sigma**2))) / (math.sqrt(2 * math.pi) * sigma)

	return kernel
