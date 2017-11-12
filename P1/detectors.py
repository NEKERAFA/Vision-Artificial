'''
	4. Detectores Bordes, Esquinas y puntos característicos
'''

import math
import numpy as np
from filters import convolve, gaussianFilter2D
from progress import *
from timing import *

'''
	Kernels para primeras derivadas en x e y
'''

def robertsKern():
	gx = np.array([[-1, 0], [0, 1]])
	gy = np.array([[0, -1], [1, 0]])
	return gx, gy

def centralDiffKern():
	gx = np.array([[-1, 0, 1]])
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
		Implementar una función que permita obtener las componentes de
		gradiente Gx y Gy de una imagen, pudiendo elegir entre los operadores de
		Roberts, CentralDiff (Diferencias centrales de Prewitt/Sobel sin
		promedio), Prewitt y Sobel.
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
		print(">> Método no reconocido " + mode)

	return [gx, gy]

def discretizacion(orientacion):
	'''
		Dado una matriz de orientaciones entre [-pi, pi], las discretiza en
		(0, 45, 90, 135)
	'''

	# Obtengo las dimensiones de la matriz de orientación
	filas, columnas = orientacion.shape

	for i in range(0, filas):
		for j in range(0, columnas):
			# Feedback
			progress(i*filas+j, filas*columnas, 'Discretizando...')

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

def vecinos_normal(magnitud, orientacion, i, j):
	'''
		Devuelve la magnitud de los vecinos que están en la normal de un punto
		borde.
	'''

	# Obtengo las dimensiones de la matriz de magnitud que debe ser igual a orientacion
	filas, columnas = magnitud.shape
	# Vecinos por defecto
	n1, n2 = -1, -1

	# Obtener vecinos a la normal del borde
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

def supresionNoMaximaGradiente(magnitud, orientacion):
	'''
		Realiza una supresión no máxima sobre un gradiente borde
	'''

	# Obtengo las dimensiones de la matriz de magnitud que debe ser igual a orientacion
	filas, columnas = magnitud.shape
	# Creo la matriz de máximos y de puntos borde
	puntos_borde = []
	maximos = magnitud.copy()

	for i in range(0, filas):
		for j in range(0, columnas):
			# Feedback
			progress(i*filas+j, filas*columnas)
			# Magnitud del pixel actual
			mag = magnitud[i, j]
			# Obtener pizel anterior y siguiente
			n1, n2 = vecinos_normal(magnitud, orientacion, i, j)
			# Asignar nuevo valor
			if mag < n1 or mag < n2:
				maximos[i, j] = 0
			else:
				puntos_borde.append((i, j))
	print()
	return puntos_borde, maximos

def vecinos_perpendicular(magnitud, orientacion, i, j):
	'''
		Devuelve las posiciones de los vecinos que están en la perpendicular a
		la normal de un punto borde
	'''

	# Obtengo las dimensiones de la matriz de magnitud que debe ser igual a orientacion
	filas, columnas = magnitud.shape
	# Valores por defecto
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

def histeresis(magnitud, orientacion, puntos_borde, bajo, alto):
	'''
		Realiza una histéresis sobre la lista de puntos bordes
	'''

	if bajo > alto:
		print("> Error, el umbral bajo es mayor que el alto")
		exit(-1)

	# Obtengo las dimensiones de la matriz de magnitud que debe ser igual a orientacion
	filas, columnas = magnitud.shape
	# Lista de bordes finales y puntos visitados
	borde = []
	visitados = np.zeros([filas, columnas])

	# Recorremos la lista de máximos (El siguiente borde)
	for (i, j) in puntos_borde:
		# print(i, j, visitados[i, j])
		# Si el borde no se ha visitado y es mayor que el umbral alto
		if visitados[i, j] == 0 and magnitud[i, j] > alto:
			# Lo visitamos y marcamos como borde
			visitados[i, j] = 1; borde.append((i, j))
			# Obtener los vecinos y añadirlos a la lista de candidatos conectados
			n1, n2 = vecinos_perpendicular(magnitud, orientacion, i, j)
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
						borde.append((i1, j1))
						# Obtenemos los vecinos y se añaden a candidatos
						n1, n2 = vecinos_perpendicular(magnitud, orientacion, i1, j1)
						candidatos.append(n1)
						candidatos.append(n2)
	return borde

def listToImage(points, img, mode):
	'''
		Dada una lista de puntos caracteristicos, lo transforma en una
		representación en matriz
	'''

	ret = None
	p, n_list = 0, len(points)

	if mode == 'points':
		ret = np.zeros(img.shape)
		for (i, j) in points:
			# Feedback
			progress(p, n_list, 'Creando...')
			p += 1
			ret[i, j] = 1
		print()
	elif mode == 'over':
		ret = np.empty([img.shape[0], img.shape[1], 3], dtype=np.uint8)
		ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = img
		for (i, j) in points:
			# Feedback
			progress(p, n_list, 'Creando...')
			p += 1
			ret[i, j, 0] = ret[i, j, 2] = 0
			ret[i, j, 1] = 192
		print()
	elif mode == 'color':
		ret = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
		ret[:, :, 1] = img
		for (i, j) in points:
			# Feedback
			progress(p, n_list, 'Creando...')
			p += 1
			ret[i, j, 1] = ret[i, j, 2] = 0
			ret[i, j, 0] = 255
		print()
	else:
		print('>> Mode not defined: ' + mode)
		exit(-1)

	return ret

@timing
def edgeCanny(inputImage , sigma , tlow , thigh, mode):
	'''
		Implementar el detector de bordes de Canny mediante una función que
		permita especificar el valor de σ en el suavizado Gaussiano y los
		umbrales del proceso de histéresis
	'''

	if tlow > thigh:
		print("> Error, el umbral bajo es mayor que el alto")
		exit(-1)

	# 1. Suavizado
	print('> 1. Suavizado')
	suavizado = gaussianFilter2D(inputImage, sigma)

	# 2. Detección de bordes
	print('> 2. Cálculo de las derivadas')
	[gx, gy] = derivatives(suavizado, 'Sobel')

	# Se calcula la magnitud
	print('> 2.1 Cálculo de la magnitud y orientación')
	magnitud = np.sqrt((gx ** 2) + (gy ** 2))

	# Se calcula la orientación
	orientacion = np.arctan2(gy, gx)

	# 3. Supresión no máxima
	print('> 3.1 Discretización de ángulos')
	discretizacion(orientacion)

	# Descarta aquellos pixeles cuyas magnitudes no alcancen ese máximo
	print('> 3.2 Supresión no máxima')
	puntos_borde, maximos = supresionNoMaximaGradiente(magnitud, orientacion)

	# 4. Histéresis
	print('> 4. Histéresis')
	lista_borde = histeresis(maximos, orientacion, puntos_borde, tlow, thigh)

	# 5. Creo la matriz de bordes
	print('> 5. Creando representación final')
	bordes = listToImage(lista_borde, inputImage, mode)

	return bordes, gx, gy, magnitud, orientacion, maximos

def matrizHarris(dx, dy, sigmaI, k):
	'''
		Crea la matriz M de Harriss
	'''
	print('>> Ixx, Iyy, Ixy')
	ixx = dx * dx
	iyy = dy * dy
	ixy = dx * dy

	print('>> Sxx')
	sxx = gaussianFilter2D(ixx, sigmaI)
	print('>> Syy')
	syy = gaussianFilter2D(iyy, sigmaI)
	print('>> Sxy')
	sxy = gaussianFilter2D(ixy, sigmaI)

	print('>> Cálculo de M')
	# Determinante del tensor [sxx sxy, sxy syy]
	det = (sxx * syy) - (sxy * sxy)
	# Traza del tensor [sxx sxy, sxy syy]
	trace = sxx + syy
	return ixx, iyy, ixy, sxx, syy, sxy, det - k * (trace * trace)

def umbralizacion(m, t):
	'''
		Umbraliza la matriz M
	'''
	filas, columnas = m.shape
	candidatos = []
	for i in range(0, filas):
		for j in range(0, columnas):
			# Feedback
			progress(i*filas+j, filas*columnas)
			# Si supera el umbral, se mete en la lista de candidatos
			if m[i, j] > t:
				candidatos.append((i, j))
	print()
	return candidatos

def supresionNoMaximaVecindario(m, candidatos, vecindario):
	'''
		Realiza una supresión no máxima sobre un vecindario de esquinas
	'''

	esquinas = []
	visitados = np.zeros(m.shape)
	ved_mid = vecindario // 2
	filas, columnas = m.shape

	p, n_candidatos = 0, len(candidatos)
	for (i, j) in candidatos:
		# Feedback
		progress(p, n_candidatos)
		p += 1

		# Obtengo las dimensiones del vecindario
		imin = max(i-ved_mid, 0)
		imax = min(i+ved_mid+1, filas-1)
		jmin = max(j-ved_mid, 0)
		jmax = min(j+ved_mid+1, columnas-1)

		vecinos = m[imin:imax, jmin:jmax]

		# Miro si el punto central es el máximo del vecindario y no hay ningún vecino visitado
		if math.isclose(m[i, j], vecinos.max()) and visitados[imin:imax, jmin:jmax].sum() == 0:
			visitados[i, j] = 1
			esquinas.append((i, j))
	print()

	return esquinas

@timing
def cornerHarris(inputImage, sigmaD, sigmaI, t, mode):
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
	print('> 1.1. Cálculo de las derivadas')
	[gx, gy] = derivatives(suavizado, 'Sobel')

	# 2. Cálculo de la matriz m
	print('> 2. Cálculo de la matriz M')
	ixx, iyy, ixy, sxx, syy, sxy, m = matrizHarris(gx, gy, sigmaI, 0.05)

	# Aplicar umbral
	print('> 3. Aplicando umbral')
	candidatos = umbralizacion(m, t)

	# Descarta aquellos pixeles cuyas magnitudes no alcancen ese máximo
	print('> 4. Aplicando supresion no máxima')
	lista_esquinas = supresionNoMaximaVecindario(m, candidatos, 3)

	# 5. Creo la matriz de bordes
	print('> 5. Creando representación final')
	esquinas = listToImage(lista_esquinas, inputImage, mode)

	print('\n> Esquinas marcadas: ', len(lista_esquinas))

	return esquinas, gx, gy, ixy, ixx, iyy, sxx, syy, sxy, m, candidatos
