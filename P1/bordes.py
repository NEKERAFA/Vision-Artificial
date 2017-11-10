'''
	4. Detectores Borde - Esquina
'''

import math
import numpy as np
from progress import progress
from filtrado import convolve, gaussianFilter2D

'''
	4. Bordes y Esquinas
'''

def robertsKern():
	gx = np.array([[-1, 0], [0, 1]])
	gy = np.array([[0, -1], [1, 0]])
	return gx, gy

def centralDiffKern():
	gx = np.array([-1, 0, 1])
	gy = gx.T
	return gx, gy

def prewittKern():
	gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
	gy = gx.T
	return gx, gy

def sobelKern():
	gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
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

def discretizacion(orientacion, filas, columnas):
	for i in range(0, filas):
		for j in range(0, columnas):
			# Feedback
			progress(i*filas+j, filas*columnas)

			# Obtiene el ángulo
			angulo = orientacion[i, j]

			# Discretiza el ángulo
			if angulo < -7*math.pi/8:
				orientacion[i, j] = 0
			elif angulo >= -7*math.pi/8 and angulo < -5*math.pi/8:
				orientacion[i, j] = 135
			elif angulo >= -5*math.pi/8 and angulo < -3*math.pi/8:
				orientacion[i, j] = 90
			elif angulo >= -3*math.pi/8 and angulo < -math.pi/8:
				orientacion[i, j] = 45
			elif angulo >= -math.pi/8 and angulo < math.pi/8:
				orientacion[i, j] = 0
			elif angulo >= math.pi/8 and angulo < 3*math.pi/8:
				orientacion[i, j] = 135
			elif angulo >= 3*math.pi/8 and angulo < 5*math.pi/8:
				orientacion[i, j] = 90
			elif angulo >= 5*math.pi/8 and angulo < 7*math.pi/8:
				orientacion[i, j] = 45
			elif angulo >= 7*math.pi/8:
				orientacion[i, j] = 0
	print()

'''
def vecino1_n(magnitud, orientacion, filas, columnas, i, j):
	if orientacion[i, j] == 0 and j > 0:
		return magnitud[i, j-1]
	elif orientacion[i, j] == 45 and j > 0 and i < filas-1:
		return magnitud[i+1, j-1]
	elif orientacion[i, j] == 90 and i < filas-1:
		return magnitud[i+1, j]
	elif orientacion[i, j] == 135 and i < filas-1 and j < columnas-1:
		return magnitud[i+1, j+1]
	else:
		return 0

def vecino2_n(magnitud, orientacion, filas, columnas, i, j):
	if orientacion[i, j] == 0 and j < columnas-1:
		return magnitud[i, j+1]
	elif orientacion[i, j] == 45 and j < columnas-1 and i > 0:
		return magnitud[i-1, j+1]
	elif orientacion[i, j] == 90 and i > 0:
		return magnitud[i-1, j]
	elif orientacion[i, j] == 135 and i > 0 and j > 0:
		return magnitud[i-1, j-1]
	else:
		return 0
'''

def vecinos_normales(magnitud, orientacion, filas, columnas, i, j):
	n1, n2 = 0, 0
	# Obtener vecinos a la normal
	if orientacion[i, j] == 0:
		n1 = magnitud[i, j-1] if j > 0 else 0
		n2 = magnitud[i, j+1] if j < columnas-1 else 0
	elif orientacion[i, j] == 45:
		n1 = magnitud[i+1, j-1] if j > 0 and i < filas-1 else 0
		n2 = magnitud[i-1, j+1] if j < columnas-1 and i > 0 else 0
	elif orientacion[i, j] == 90:
		n1 = magnitud[i+1, j] if i < filas-1 else 0
		n2 = magnitud[i-1, j] if i > 0 else 0
	elif orientacion[i, j] == 135:
		n1 = magnitud[i+1, j+1] if i < filas-1 and j < columnas-1 else 0
		n2 = magnitud[i-1, j-1] if i > 0 and j > 0 else 0
	return n1, n2

def supresionNoMaximaGradiente(magnitud, orientacion, filas, columnas):
	bordesmaximos = []
	maximos = magnitud.copy()
	for i in range(0, filas):
		for j in range(0, columnas):
			# Feedback
			progress(i*filas+j, filas*columnas)
			# Magnitud del pixel actual
			mag = magnitud[i, j]
			# Obtener pizel anterior y siguiente
			n1, n2 = vecinos_normales(magnitud, orientacion, filas, columnas, i, j)
			# Asignar nuevo valor
			if mag < n1 or mag < n2:
				maximos[i, j] = 0
			else:
				bordesmaximos.append((i, j))
	print()
	return bordesmaximos, maximos

def vecinos_perpendiculares(magnitud, orientacion, filas, columnas, i, j):
	i1, j1, i2, j2 = -1, -1, -1, -1
	# Obtener vecinos a la perpendicular de la normal (Por el borde)
	if orientacion[i, j] == 90:
		if j > 0:
			i1, j1 = i, j-1
		if j < columnas-1:
			i2, j2 = i, j+1
	elif orientacion[i, j] == 135:
		if j > 0 and i < filas-1:
			i1, j1 = i+1, j-1
		if j < columnas-1 and i > 0:
			i2, j2 = i-1, j+1
	elif orientacion[i, j] == 0:
		if i < filas-1:
			i1, j1 = i+1, j
		if i > 0:
			i2, j2 = i-1, j
	elif orientacion[i, j] == 45:
		if i < filas-1 and j < columnas-1:
			i1, j1 = i+1, j+1
		if i > 0 and j > 0:
			i2, j2 = i-1, j-1
	return (i1, j1), (i2, j2)

def histeresis(magnitud, orientacion, maximos, bajo, alto, filas, columnas):
	if bajo > alto:
		print("> Error, el umbral bajo es mayor que el alto")
		exit(-1)

	# Matriz de bordes finales y puntos visitados
	borde = np.zeros([filas, columnas])
	visitados = np.zeros([filas, columnas])
	# Recorremos la lista de máximos (El siguiente borde)
	for (i, j) in maximos:
		# print(i, j, visitados[i, j])
		# Si el borde no se ha visitado y es mayor que el umbral alto
		if visitados[i, j] == 0 and magnitud[i, j] > alto:
			# Lo visitamos y marcamos como borde
			visitados[i, j] = 1; borde[i, j] = 1
			# Obtener los vecinos y añadirlos a la lista de candidatos conectados
			n1, n2 = vecinos_perpendiculares(magnitud, orientacion, filas, columnas, i, j)
			candidatos = [n1, n2]
			# Vamos sacando los candidatos hasta que no haya
			while len(candidatos):
				# Sacamos un candidato de la lista
				(i1, j1) = candidatos.pop(0)
				# Si no ha sido visitado, lo marcamos como visitado
				if i1 >= 0 and j1 >= 0 and visitados[i1, j1] == 0:
					visitados[i1, j1] = 1
					# Si el candidato es mayor que el umbral bajo lo añadimos como borde
					if magnitud[i1, j1] > bajo:
						borde[i1, j1] = 1
						# Obtenemos los vecinos y se añaden a candidatos
						n1, n2 = vecinos_perpendiculares(magnitud, orientacion, filas, columnas, i1, j1)
						candidatos.append(n1)
						candidatos.append(n2)
	return borde

def edgeCanny(inputImage , sigma , tlow , thigh):
	'''
		Implementar el detector de bordes de Canny mediante una función que
		permita especificar el valor de σ en el suavizado Gaussiano y los
		umbrales del proceso de histéresis
	'''

	if tlow > thigh:
		print("> Error, el umbral bajo es mayor que el alto")
		exit(-1)

	# Obtengo las dimensiones
	filas, columnas = inputImage.shape[0], inputImage.shape[1]

	# 1. Suavizado
	print('> 1. Suavizado')
	suavizado = gaussianFilter2D(inputImage, sigma)

	# 2. Detección de bordes
	print('> 2. Detección de bordes')
	[gx, gy] = derivatives(suavizado, 'Sobel')

	# Se calcula la magnitud
	print('> 2.1 Cálculo de la magnitud')
	magnitud = np.sqrt((gx ** 2) + (gy ** 2))

	# Se calcula la orientación
	print('> 2.2 Cálculo de la orientación')
	orientacion = np.arctan2(gy, gx)

	# 3. Supresión no máxima
	print('> 3.1 Discretización de ángulos')
	discretizacion(orientacion, filas, columnas)

	# Descarta aquellos pixeles cuyas magnitudes no alcancen ese máximo
	print('> 3.2 Supresión no máxima')
	bordesmaximos, maximos = supresionNoMaximaGradiente(magnitud, orientacion, filas, columnas)

	# 4. Histéresis
	print('> 4. Histéresis')
	borde = histeresis(maximos, orientacion, bordesmaximos, tlow, thigh, filas, columnas)

	return borde, gx, gy, magnitud, orientacion, maximos


def cornerHarris(inputImage, sigmaD, sigmaI, t, k=0.05):
	'''
		Implementar el detector de esquinas de Harris que utilice Gaussianas
		tanto para la diferenciación como para la integración. La función
		permitirá establecer la escala de diferenciación σ D, la escala de
		integración σ I , y el valor del umbral para esquinas (t)
	'''

	# Obtengo las dimensiones
	filas, columnas = inputImage.shape[0], inputImage.shape[1]

	# 1.1 Suavizado
	print('> 1.1 Suavizado')
	suavizado = gaussianFilter2D(inputImage, sigmaD)

	# 1.2 Cálculo de las derivadas
	print('> 1.1. Se calculan las derivadas')
	[gx, gy] = derivatives(suavizado, 'Sobel')

	# 2.1 Se calcula los elementos de la Matriz
	print('> 2.1 Cálculo los elementos de la matriz')
	print('> 2.1.1 Ixx y Sxx')
	ixx = gx * gx
	sxx = gaussianFilter2D(ixx, sigmaI)
	print('> 2.1.2 Iyy y Syy')
	iyy = gy * gy
	syy = gaussianFilter2D(iyy, sigmaI)
	print('> 2.1.3 Ixy y Sxy')
	ixy = gx * gy
	sxy = gaussianFilter2D(ixy, sigmaI)

	# 2.2 Se computa harrys
	print('> 2.2 Cálculo del determinante')
	det = np.zeros([filas, columnas])
	det = (sxx * syy) - (sxy * sxy)

	# 2.3 Se computa harrys
	print('> 2.3 Cálculo de la traza')
	trace = sxx + syy

	print('> 2.4 Cálculo de harrys')
	m = det - k * (trace * trace)

	# Aplicar umbral
	print('> 3. Aplicando umbral')
	candidatos = []
	for i in range(0, filas):
		for j in range(0, columnas):
			progress(i*filas+j, filas*columnas)
			if m[i, j] > t:
				candidatos.append((i, j))

	# Descarta aquellos pixeles cuyas magnitudes no alcancen ese máximo
	print('> 4. Aplicando supresion no máxima')
	esquinas = np.empty([filas, columnas, 3], dtype=np.uint8)
	esquinas[:, :, 2] = esquinas[:, :, 1] = esquinas[:, :, 0] = inputImage

	visitados = np.zeros([filas, columnas])
	n = n_esquinas = 0
	for (i, j) in candidatos:
		# Feedback
		progress(n, len(candidatos))
		n += 1
		imin = max(i-1, 0)
		imax = min(i+2, filas-1)
		jmin = max(j-1, 0)
		jmax = min(j+2, columnas-1)
		vecinos = m[imin:imax, jmin:jmax]
		if math.isclose(m[i, j], vecinos.max()) and visitados[imin:imax, jmin:jmax].sum() == 0:
			n_esquinas += 1
			esquinas[i, j, 2] = esquinas[i, j, 1] = 0
			esquinas[i, j, 0] = 255
			visitados[i, j] = 1

	print('\n> Esquinas marcadas: ', n_esquinas, '\n')
	return esquinas, gx, gy, ixy, ixx, iyy, sxx, syy, sxy, m, candidatos
