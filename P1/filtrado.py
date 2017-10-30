'''
	2. Funciones de suavizado y realce
'''

from math import *
import numpy as np

def convolve( inputImage, kernel ):
	'''
		Permite realizar un filtrado espacial sobre una imagen con un kernel
		arbitrario que se pasará por parámetro.
	'''

	# Obtengo las dimensiones
	width, height = inputImage.shape[0], inputImage.shape[1]
	width_ker, height_ker = kernel.shape[0], kernel.shape[1]
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
						value = value + inputImage[x, y] * kernel[k, l]

			outputImage[i, j] = value

	return outputImage

def gaussKernel1D( sigma ):
	'''
		Calcula un kernel Gaussiano horizontal 1 × N , a partir de σ que será
		pasado como parámetro, y calculando N como N = 2 ceil|3σ| + 1.
	'''

	# Obtengo la dimensión
	n = 2 * ceil(3 * sigma) + 1
	# Creo el kernel de salida
	kernel = np.zeros([1, n])

	for i in range(0, n):
		x = i - (n // 2)
		kernel[0, i] = (1 / (sqrt(2 * pi) * sigma)) * exp((-x**2) / (2 * (sigma**2)))

	return kernel

def gaussianFilter2D( inputImage, sigma ):
	'''
		Permite realizar un suavizado Gaussiano bidimensional usando un filtro
		N×N de parámetro σ, donde N se calcula igual que en la función anterior.
	'''

	kernel = gaussKernel1D(sigma)
	outputImage = convolve(inputImage, kernel)
	outputImage = convolve(outputImage, kernel.T)

	return outputImage

def medianFilter2D( inputImage, filterSize ):
	'''
		Implementa el filtro de orden de medianas. Permitire establecer el
		tamaño del filtro.
	'''

	# Obtengo las dimensiones
	width, height = inputImage.shape[0], inputImage.shape[1]
	# Creo la imagen de salida
	outputImage = np.zeros([width, height])

	# Recorro los píxeles de la imagen
	for i in range(0, width):
		for j in range(0, height):
			x_min = max(i - floor(filterSize / 2), 0)
			x_max = min(i + ceil(filterSize / 2), width)
			y_min = max(j - floor(filterSize / 2), 0)
			y_max = min(j + ceil(filterSize / 2), height)

			outputImage[i, j] = np.median(inputImage[x_min:x_max, y_min:y_max])

	return outputImage

def highBoost( inputImage, A, method, parameter ):
	'''
		Permite especificar, además del factor de amplificación A, el método de
		suavizado utilizado y su parámetro. Las opciones serán filtrado
		Gaussiano (con parámetro σ) y filtrado de medianas (con tamaño de
		ventana como parámetro).

		method = 'gaussian' | 'median'
	'''

	# Obtengo la imagen suavizada
	smooth = None

	if method == 'gaussian':
		smooth = gaussianFilter2D(inputImage, parameter)
	elif method == 'median':
		smooth = medianFilter2D(inputImage, parameter)
	else:
		print("> Método no reconocido")

	# Obtengo las dimensiones
	width, height = inputImage.shape[0], inputImage.shape[1]
	# Creo la imagen de salida
	'''outputImage = np.zeros([width, height])

	# Recorro los píxeles de la imagen

	for i in range(0, width):
		for j in range(0, height):
			outputImage[i][j] = A * inputImage[i][j] - smooth[i][j]
	'''
	return A * inputImage - smooth

	#return outputImage
