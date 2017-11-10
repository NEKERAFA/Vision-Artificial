#!/usr/bin/python3

'''
	Archivo con funciones de ejemplo
'''

import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import sys, getopt, math, time
from progress import progress

import histograma
import filtrado
import operadores
import bordes

# Obtiene más información en los operadores grandes (Canny y Harrys)
debug = False
# Último plot hecho
plot = 0

def showHist(image, title=None, vmin=0, vmax=255):
	'''
		Dibuja un plot con un histograma dado una imagen
	'''
	# Creo un nuevo plot
	global plot
	plot = plot+1
	plt.figure(plot)
	# Muestro el histograma de la imagen
	plt.hist(image.ravel(), range(vmin, vmax + 1))
	if title is not None:
		plt.title(title)

def showImg(image, title=None, vmin=0, vmax=255):
	'''
		Dibuja un plot con una imagen
	'''
	# Creo un nuevo plot
	global plot
	plot = plot+1
	plt.figure(plot)
	# Muestro la imagen
	plt.imshow(image, cmap = 'gray', vmin = vmin, vmax = vmax)
	if title is not None:
		plt.title(title)

def Input(path, hist):
	'''
		Carga una imagen y/o muestra su histograma
	'''
	# Cargo una imagen y la muestra
	original = misc.imread(path, True)
	showImg(original, 'Imagen de entrada')
	# Muestra el histograma si procediese
	if hist:
		showHist(original, 'Histograma de entrada')
	return original

def InputBinary(path, hist):
	'''
		Carga una imagen y la convierte en binaria
	'''
	original = misc.imread(path, True)
	# Recorro la imagen para dejarla en 0 o 1
	for i in range(0, original.shape[0]):
		for j in range(0, original.shape[1]):
			original[i][j] = 0 if original[i][j] < 128 else 1
	showImg(original, 'Imagen de entrada', 0, 1)
	# Muestra el histograma si procediese
	if hist:
		showHist(original, 'Histograma de entrada', 0, 1)
	return original

def Output(output, hist, path):
	'''
		Muestra una imagen de salida y/o la guarda
	'''
	# Guarda una imagen si pocediese
	if path is not None:
		misc.imsave(path + '.png', output)
	# Muestra su histograma si procediese
	if hist:
		showHist(output, 'Histograma de salida')
	# Muestra la imagen
	showImg(output, 'Imagen de salida')

def OutputBinary(output, hist, path):
	'''
		Muestra una imagen binaria y/o la guarda si procede
	'''
	# Guarda una imagen si pocediese
	if path is not None:
		misc.imsave(path + '.png', output)
	# Muestra su histograma si procediese
	if hist:
		showHist(output, 'Histograma de salida', 0, 1)
	# Muestra la imagen
	showImg(output, 'Imagen de salida', 0, 1)

def OutputOperator(output, title, hist, path):
	'''
		Muestra una imagen procedente de un operador ajustando el contraste y/o la guarda si procede
	'''
	# Guarda la imagen si procediese
	if path is not None:
		misc.imsave(path + '.png', output)
	# Muestra su histograma si procediese
	if hist:
		showHist(output, 'Histograma de ' + title, int(output.min()), int(output.max()))
	# Muestra la imagen
	showImg(output, title, output.min(), output.max())

def histEnhance(pathInput, center, win, hist, nameOutput):
	win = abs(win)
	print("> Realce de contraste 'window-level' centrado en " + str(center) + " con ventana " + str(win))
	original = Input(pathInput, hist)
	output = histograma.histEnhance(original, center, win)
	print("> Hecho")
	Output(output, hist, nameOutput)
	plt.show()

def histAdapt(pathInput, minValue, maxValue, hist, nameOutput):
	if minValue > maxValue:
		print("> El valor mínimo tiene que ser más pequeño que el valor máximo")
	print("> Compresion/estiramiento de histograma con valores [" + str(minValue) + ", " + str(maxValue) + "]")
	original = Input(pathInput, hist)
	output = histograma.histAdapt(original, minValue, maxValue)
	print("> Hecho")
	Output(output, hist, nameOutput)
	plt.show()

def convolve(pathInput, pathKernel, hist, nameOutput):
	print("> Prueba de convolución")
	original = Input(pathInput, hist)
	kernel = misc.imread(pathKernel, True)
	output = filtrado.convolve(original, kernel)
	print("> Hecho")
	Output(output, hist, nameOutput)
	plt.show()

def gaussian(pathInput, sigma, hist, nameOutput):
	print("> Filtro Gaussiano con sigma " + str(sigma))
	original = Input(pathInput, hist)
	time1 = time.time()
	output = filtrado.gaussianFilter2D(original, sigma)
	time2 = time.time()
	print('> Hecho en {:0.2f} ms'.format((time2-time1)*1000.0))
	Output(output, hist, nameOutput)
	plt.show()

def median(pathInput, filterSize, hist, nameOutput):
	print("> Filtro de medianas con tamaño " + str(filterSize))
	original = Input(pathInput, hist)
	output = filtrado.medianFilter2D(original, filterSize)
	print("> Hecho")
	Output(output, hist, nameOutput)
	plt.show()

def highBoost(pathInput, A, method, parameter, hist, nameOutput):
	print("> High Boost con amplificación " + str(A) + " y método " + method + "(" + str(parameter) + ")")
	original = Input(pathInput, hist)
	output = filtrado.highBoost(original, A, method, parameter)
	print("> Hecho")
	output = histograma.histAdapt(output, 0, 255)
	Output(output, hist, nameOutput)
	plt.show()

def dilate(pathInput, ElType, size, hist, nameOutput):
	print("> Dilatación con EE " + ElType + " de tamaño " + str(size))
	original = InputBinary(pathInput, hist)
	output = operadores.dilate(original, ElType, size)
	print("> Hecho")
	OutputBinary(output, hist, nameOutput)
	plt.show()

def erode(pathInput, ElType, size, hist, nameOutput):
	print("> Erosión con EE " + ElType + " de tamaño " + str(size))
	original = InputBinary(pathInput, hist)
	output = operadores.erode(original, ElType, size)
	print("> Hecho")
	OutputBinary(output, hist, nameOutput)
	plt.show()

def opening(pathInput, ElType, size, hist, nameOutput):
	print("> Apertura con EE " + ElType + " de tamaño " + str(size))
	original = InputBinary(pathInput, hist)
	output = operadores.opening(original, ElType, size)
	print("> Hecho")
	OutputBinary(output, hist, nameOutput)
	plt.show()

def closing(pathInput, ElType, size, hist, nameOutput):
	print("> Cierre con EE " + ElType + " de tamaño " + str(size))
	original = InputBinary(pathInput, hist)
	output = operadores.closing(original, ElType, size)
	print("> Hecho")
	OutputBinary(output, hist, nameOutput)
	plt.show()

def tophatFilter(pathInput, ElType, size, mode, hist, nameOutput):
	print("> Filtro Top-Hat de modo " + mode + " con EE " + ElType + " de tamaño " + str(size))
	original = InputBinary(pathInput, hist)
	output = operadores.tophatFilter(original, ElType, size, mode)
	print("> Hecho")
	OutputBinary(output, hist, nameOutput)
	plt.show()

def derivatives(pathInput, operator, hist, nameOutput):
	print("> Derivadas primeras con " + operator)
	original = Input(pathInput, hist)
	[gx, gy] = bordes.derivatives(original, operator)
	print("> Hecho")
	# Si hay nombre de salida, genera nuevos nombres para las derivadas
	gx_name, gy_name = None, None
	if nameOutput is not None:
		gx_name = nameOutput + '_Gx'
		gy_name = nameOutput + '_Gy'
	OutputOperator(gx, 'Gx', hist, gx_name)
	OutputOperator(gy, 'Gy', hist, gy_name)
	plt.show()

'''
def listToImage(l, rows, cols, magnitude=None):
	img = np.zeros([rows, cols])
	len_list = len(l)

	pix = 1
	for (i, j) in l:
		# Feedback
		progress(pix, len_list, 'Creando ...')
		# Pone el valor de la lista
		if magnitude is None:
			img[i, j] = 1
		else:
			img[i, j] = magnitude[i, j]
		pix += 1

	print()
	return img
'''

def edgeCanny(pathInput, sigma, tlow, thigh, hist, nameOutput):
	if tlow > thigh:
		print("> Operador Canny: Error, el umbral bajo es mayor que el alto")
		sys.exit(-1)

	print("> Operador Canny con sigma " +  str(sigma) + ", umbral menor " + str(tlow) + " y umbral mayor " + str(thigh))
	original = Input(pathInput, False)
	border, gx, gy, magnitude, orientation, max_border = bordes.edgeCanny(original, sigma, tlow, thigh)
	print("> Hecho")

	# Salidas para debug, estados intermedios
	if debug:
		gx_name, gy_name, mag_name, or_name, max_name = None, None, None, None, None
		if nameOutput is not None:
			gx_name  = nameOutput + '_gx'
			gy_name  = nameOutput + '_gy'
			mag_name = nameOutput + '_m'
			or_name  = nameOutput + '_o'
			max_name = nameOutput + '_s'

		OutputOperator(gx, 'Gx', hist, gx_name)
		OutputOperator(gy, 'Gy', hist, gy_name)
		OutputOperator(magnitude, 'Magnitud', hist, mag_name)
		OutputOperator(orientation, 'Orientacion', hist, or_name)
		OutputOperator(max_border, 'Supresion no máxima', hist, max_name)

	# Salida de Canny
	OutputOperator(border, 'Canny', hist, nameOutput)
	plt.show()

def cornerHarris(pathInput, sigmaD, sigmaI, t, hist, nameOutput, k=0.06):
	print('> Operador Harris con sigmaD ' + str(sigmaD) + ', sigmaI ' + str(sigmaI) + ' y umbral ' + str(t))
	original = Input(pathInput, hist)
	edges, gx, gy, ixy, ixx, iyy, sxx, syy, sxy, m, candidatos = bordes.cornerHarris(original, sigmaD, sigmaI, t, k)
	print("> Hecho")

	# Salidas para debug
	if debug:
		# Crear imagen de candidatos
		print("> Creando imagen de candidatos")
		umbral = np.zeros([original.shape[0], original.shape[1]])
		for (i, j) in candidatos:
			umbral[i, j] = 1

		gx_n, gy_n, ixx_n, iyy_n, ixy_n, sxx_n, syy_n, sxy_n, m_n, c_n = None, None, None, None, None, None, None, None, None, None
		if nameOutput is not None:
			gx_n  = nameOutput + '_gx'
			gy_n  = nameOutput + '_gy'
			ixx_n = nameOutput + '_ixx'
			iyy_n = nameOutput + '_iyy'
			ixy_n = nameOutput + '_ixy'
			sxx_n = nameOutput + '_sxx'
			syy_n = nameOutput + '_syy'
			sxy_n = nameOutput + '_sxy'
			m_n   = nameOutput + '_m'
			c_n   = nameOutput + '_candidatos'

		OutputOperator(gx, 'Gx', hist, gx_n)
		OutputOperator(gy, 'Gy', hist, gy_n)
		OutputOperator(ixx, 'Ixx', hist, ixx_n)
		OutputOperator(iyy, 'Iyy', hist, iyy_n)
		OutputOperator(ixy, 'Ixy', hist, ixy_n)
		OutputOperator(sxx, 'Sxx', hist, sxx_n)
		OutputOperator(syy, 'Syy', hist, syy_n)
		OutputOperator(sxy, 'Sxy', hist, sxy_n)
		OutputOperator(m, 'Matriz M', hist, m_n)
		OutputOperator(umbral, 'Umbral de esquinas', hist, c_n)

	# Salida de Harrys
	OutputOperator(edges, 'Esquinas', hist, nameOutput)
	plt.show()

def main(argv):
	inputPath = None
	outputPath = None
	method = ''
	hist = False
	usage = ('./p1.py --input=<path> --output=<name> --histogram --debug --method=<method> <args>\n\n'
	         'Methods and arguments:\n'
			 '   --method=histEnhance <cenValue> <winSize>\n'
			 '   --method=histAdapt <minValue> <maxValue>\n'
			 '   --method=convolve <pathKernel>\n'
			 '   --method=gaussian <sigma>\n'
			 '   --method=median <filterSize>\n'
			 '   --method=highBoost <A> <method> <parameter>\n'
			 '   --method=dilate <ElType> <size>\n'
			 '   --method=erode <ElType> <size>\n'
			 '   --method=opening <ElType> <size>\n'
			 '   --method=closing <ElType> <size>\n'
			 '   --method=topHat <ElType> <size> <mode>\n'
			 '   --method=derivatives <operator>\n'
			 '   --method=canny <sigma> <tlow> <thigh>\n'
			 '   --method=harris <sigmaD> <sigmaI> <t>')

	try:
		opts, args = getopt.getopt(argv, "i:o:hdm:", ["method=", "input=", "output=", "histogram", "debug"])
	except getopt.GetoptError:
		print("> Error al parsear argumentos")
		print(usage)
		sys.exit(-1)

	for opt, arg in opts:
		if opt in ("-m", "--method"):
			method = arg
		elif opt in ("-i", "--input"):
			inputPath = arg
		elif opt in ("-o", "--output"):
			outputPath = arg
		elif opt in ("-h", "--histogram"):
			hist = True
		elif opt in ("-d", "--debug"):
			global debug
			debug = True

	if inputPath is None:
		print('> Introduce al menos una ruta de imagen')
		print(usage)
		sys.exit(-1)

	if method == "histEnhance" and len(args) == 2:
		histEnhance(inputPath, int(args[0]), int(args[1]), hist, outputPath)
	elif method == "histAdapt" and len(args) == 2:
		histAdapt(inputPath, int(args[0]), int(args[1]), hist, outputPath)
	elif method == "convolve" and len(args) == 1:
		convolve(inputPath, args[0], hist, outputPath)
	elif method == "gaussian" and len(args) == 1:
		gaussian(inputPath, float(args[0]), hist, outputPath)
	elif method == "median" and len(args) == 1:
		median(inputPath, int(args[0]), hist, outputPath)
	elif method == "highBoost" and len(args) == 3:
		highBoost(inputPath, float(args[0]), args[1], float(args[2]), hist, outputPath)
	elif method == "dilate" and len(args) == 2:
		dilate(inputPath, args[0], int(args[1]), hist, outputPath)
	elif method == "erode" and len(args) == 2:
		erode(inputPath, args[0], int(args[1]), hist, outputPath)
	elif method == "opening" and len(args) == 2:
		opening(inputPath, args[0], int(args[1]), hist, outputPath)
	elif method == "closing" and len(args) == 2:
		closing(inputPath, args[0], int(args[1]), hist, outputPath)
	elif method == "topHat" and len(args) == 3:
		tophatFilter(inputPath, args[0], int(args[1]), args[2], hist, outputPath)
	elif method == "derivatives" and len(args) == 1:
		derivatives(inputPath, args[0], hist, outputPath)
	elif method == "canny" and len(args) == 3:
		edgeCanny(inputPath, float(args[0]), int(args[1]), int(args[2]), hist, outputPath)
	elif method == "harris" and len(args) == 3:
		cornerHarris(inputPath, float(args[0]), float(args[1]), int(args[2]), hist, outputPath)
	elif method == "harris" and len(args) == 4:
		cornerHarris(inputPath, float(args[0]), float(args[1]), int(args[2]), hist, outputPath, k=float(args[3]))
	else:
		print("> Parámetros pasados:", inputPath, outputPath, hist, debug, method, args)
		print(usage)
		sys.exit()

if __name__ == "__main__":
	main(sys.argv[1:])
