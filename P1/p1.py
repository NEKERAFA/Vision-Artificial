#!/usr/bin/python3

'''
	Archivo con funciones de ejemplo
'''

import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import sys, getopt, math, time
from timing import *

import hists
import filters
import bin_op
import detectors

'''
	Variables globales
'''

# Imagen de entrada
inputImg = None
# Ruta de la imagen de entrada
inputPath = None
# Imagen de salida
outputImg = None
# Nombre de la imagen de salida
outputName = None
# Histograma
histogram = False
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

def Input():
	'''
		Carga la imagen de entrada y/o muestra su histograma
	'''
	global inputImg, inputPath, histogram
	# Cargo una imagen y la muestra
	inputImg = misc.imread(inputPath, True)
	showImg(inputImg, 'Imagen de entrada')
	# Muestra el histograma si procediese
	if histogram:
		showHist(inputImg, 'Histograma de entrada')

def InputBinary():
	'''
		Carga una imagen para convertirla en binaria y/o muestra su histograma
	'''
	global inputImg, inputPath, histogram
	# Cargo una imagen
	inputImg = misc.imread(inputPath, True)
	# Recorro la imagen para dejarla en 0 o 1
	for i in range(0, inputImg.shape[0]):
		for j in range(0, inputImg.shape[1]):
			inputImg[i][j] = 0 if inputImg[i][j] < 128 else 1

	showImg(inputImg, 'Imagen de entrada', 0, 1)
	# Muestra el histograma si procediese
	if histogram:
		showHist(inputImg, 'Histograma de entrada', 0, 1)

def Output():
	'''
		Muestra una imagen de salida y/o la guarda
	'''
	global outputImg, outputName, histogram
	# Guarda una imagen si pocediese
	if outputName is not None:
		misc.imsave(outputName + '.png', outputImg)
	# Muestra su histograma si procediese
	if histogram:
		showHist(outputImg, 'Histograma de salida')
	# Muestra la imagen
	showImg(outputImg, 'Imagen de salida')

def OutputBinary():
	'''
		Muestra una imagen binaria y/o la guarda si procede
	'''
	global outputImg, outputName, histogram
	# Guarda una imagen si pocediese
	if outputName is not None:
		misc.imsave(outputName + '.png', outputImg)
	# Muestra su histograma si procediese
	if histogram:
		showHist(outputImg, 'Histograma de salida', 0, 1)
	# Muestra la imagen
	showImg(outputImg, 'Imagen de salida', 0, 1)

def OutputOperator(outputImg, operator=None, title='Imagen de salida', titleHist='Histograma de salida'):
	'''
		Muestra una imagen procedente de un operador ajustando el contraste
		y/o la guarda si procede
	'''
	global outputName, histogram
	# Guarda una imagen si pocediese
	if outputName is not None:
		if operator is None:
			misc.imsave(outputName + '.png', outputImg)
		else:
			misc.imsave(outputName + '_' + operator + '.png', outputImg)
	# Muestra su histograma si procediese
	if histogram:
		showHist(outputImg, titleHist, int(outputImg.min()), int(outputImg.max()))
	# Muestra la imagen
	showImg(outputImg, title, int(outputImg.min()), int(outputImg.max()))

'''
	Funciones envoltorio
'''

def histEnhance(center, win):
	global inputImg, outputImg
	print("> Realce de contraste 'window-level' centrado en " + str(center) + " con ventana " + str(win))
	Input()
	outputImg = hists.histEnhance(inputImg, center, win)
	Output()

def histAdapt(minValue, maxValue):
	global inputImg, outputImg
	print("> Compresion/estiramiento de histograma con valores [" + str(minValue) + ", " + str(maxValue) + "]")
	Input()
	outputImg = hists.histAdapt(inputImg, minValue, maxValue)
	Output()

def convolve(pathKernel):
	global inputImg, outputImg
	print("> Prueba de convolución")
	Input()
	# Cargo el kernel
	kernel = misc.imread(pathKernel, True)
	convolve = timing(filters.convolve)
	outputImg = convolve(inputImg, kernel)
	Output()

def gaussianFilter(sigma):
	global inputImg, outputImg
	print("> Filtro Gaussiano con sigma " + str(sigma))
	Input()
	gaussianFilter = timing(filters.gaussianFilter2D)
	outputImg = gaussianFilter(inputImg, sigma)
	Output()

def medianFilter(filterSize):
	global inputImg, outputImg
	print("> Filtro de medianas con tamaño " + str(filterSize))
	Input()
	medianFilter = timing(filters.medianFilter2D)
	outputImg = medianFilter(inputImg, filterSize)
	Output()

def highBoost(A, method, parameter):
	global inputImg, outputImg, outputName
	print("> High Boost con amplificación " + str(A) + " y método " + method + "(" + str(parameter) + ")")
	Input()
	outputImg = filters.highBoost(inputImg, A, method, parameter)
	OutputOperator(outputImg, outputName, 'Imagen de salida')

def dilate(ElType, size):
	global inputImg, outputImg
	print("> Dilatación con EE " + ElType + " de tamaño " + str(size))
	InputBinary()
	dilate = timing(bin_op.dilate)
	outputImg = dilate(inputImg, ElType, size)
	OutputBinary()

def erode(ElType, size):
	global inputImg, outputImg
	print("> Erosión con EE " + ElType + " de tamaño " + str(size))
	InputBinary()
	erode = timing(bin_op.erode)
	outputImg = erode(inputImg, ElType, size)
	OutputBinary()

def opening(ElType, size):
	global inputImg, outputImg
	print("> Apertura con EE " + ElType + " de tamaño " + str(size))
	InputBinary()
	opening = timing(bin_op.opening)
	outputImg = opening(inputImg, ElType, size)
	OutputBinary()

def closing(ElType, size):
	global inputImg, outputImg
	print("> Cierre con EE " + ElType + " de tamaño " + str(size))
	InputBinary()
	closing = timing(bin_op.closing)
	outputImg = closing(inputImg, ElType, size)
	OutputBinary()

def tophatFilter(ElType, size, mode):
	global inputImg, outputImg
	print("> Filtro Top-Hat de modo " + mode + " con EE " + ElType + " de tamaño " + str(size))
	InputBinary()
	outputImg = bin_op.tophatFilter(inputImg, ElType, size, mode)
	OutputBinary()

def derivatives(operator):
	global inputImg
	print("> Derivadas de primer orden con el método " + operator)
	Input()
	derivatives = timing(detectors.derivatives)
	[gx, gy] = derivatives(inputImg, operator)
	OutputOperator(gx, 'gx', 'Gradiente en x')
	OutputOperator(gy, 'gy', 'Gradiente en y')

def edgeCanny(sigma, tlow, thigh, mode='points'):
	if tlow > thigh:
		print(">> Operador Canny: El umbral bajo es mayor que el alto")
		sys.exit(-1)

	global inputImg
	print("> Operador Canny con sigma " +  str(sigma) + ", umbral menor " + str(tlow) + " y umbral mayor " + str(thigh))
	Input()
	border, gx, gy, magnitude, orientation, max_border = detectors.edgeCanny(inputImg, sigma, tlow, thigh, mode)

	# Salidas para debug, estados intermedios
	if debug:
		OutputOperator(gx, 'gx', 'Gradiente en x')
		OutputOperator(gy, 'gy', 'Gradiente en y')
		OutputOperator(magnitude, 'm', 'Magnitud')
		OutputOperator(orientation, 'o', 'Orientacion')
		OutputOperator(max_border, 'sup', 'Supresion no máxima')

	# Salida de Canny
	OutputOperator(border, title = 'Operador Canny')
	plt.show()

def cornerHarris(sigmaD, sigmaI, t, mode='over'):
	global inputImg
	print('> Operador Harris con sigmaD ' + str(sigmaD) + ', sigmaI ' + str(sigmaI) + ' y umbral ' + str(t))
	Input()
	edges, gx, gy, ixy, ixx, iyy, sxx, syy, sxy, m, candidates = detectors.cornerHarris(inputImg, sigmaD, sigmaI, t, mode)

	# Salidas para debug
	if debug:
		# Crear imagen de candidatos
		print("> Creando imagen de candidatos")
		umbral = np.zeros(inputImg.shape)
		n, len_candidates
		for (i, j) in candidatos:
			progress(n, len_candidates, 'Creando...')
			n += 1
			umbral[i, j] = 1

		OutputOperator(gx, 'gx', 'Gradiente en x')
		OutputOperator(gy, 'gy', 'Gradiente en y')
		OutputOperator(ixy, 'ixy', 'Ixy')
		OutputOperator(ixx, 'ixx', 'Ixx')
		OutputOperator(iyy, 'iyy', 'Iyy')
		OutputOperator(sxy, 'sxy', 'Sxy')
		OutputOperator(sxx, 'sxx', 'Sxx')
		OutputOperator(syy, 'syy', 'Syy')
		OutputOperator(m, 'm', 'Matriz M')
		OutputOperator(umbral, 'c', 'Candidatos')

	# Salida de Harrys
	OutputOperator(edges, title = 'Operador Harrys')

def main(argv):
	global inputPath, outputName, histogram, debug
	method = ''
	usage = ('./p1.py --input=<path> --output=<name> --histogram --debug --method=<method> <args>\n\n'
	         'Methods and arguments:\n'
			 '\t Histogram:\n'
			 '   --method=histEnhance <cenValue> <winSize>\n'
			 '   --method=histAdapt <minValue> <maxValue>\n\n'
			 '\t Filters:\n'
			 '   --method=convolve <pathKernel>\n'
			 '   --method=gaussianFilter2D <sigma>\n'
			 '   --method=medianFilter2D <filterSize>\n'
			 '   --method=highBoost <A> <method = black | white> <parameter>\n\n'
			 '\t Morphological operators:\n'
			 '\t ElType = square | lineh | linev | cross\n'
			 '   --method=dilate <ElType> <size>\n'
			 '   --method=erode <ElType> <size>\n'
			 '   --method=opening <ElType> <size>\n'
			 '   --method=closing <ElType> <size>\n'
			 '   --method=tophatFilter <ElType> <size> <mode>\n\n'
			 '\t Detectors:\n'
			 '   --method=derivatives <operator>\n'
			 '   --method=canny <sigma> <tlow> <thigh> [output = points | over | color]\n'
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
			outputName = arg
		elif opt in ("-h", "--histogram"):
			histogram = True
		elif opt in ("-d", "--debug"):
			debug = True

	if inputPath is None:
		print('> Introduce al menos la ruta de imagen a usar')
		print(usage)
		sys.exit(-1)


	if method == "histEnhance" and len(args) == 2:
		histEnhance(int(args[0]), int(args[1]))
	elif method == "histAdapt" and len(args) == 2:
		histAdapt(int(args[0]), int(args[1]))
	elif method == "convolve" and len(args) == 1:
		convolve(args[0])
	elif method == "gaussianFilter2D" and len(args) == 1:
		gaussianFilter(float(args[0]))
	elif method == "medianFilter2D" and len(args) == 1:
		medianFilter(int(args[0]))
	elif method == "highBoost" and len(args) == 3:
		highBoost(float(args[0]), args[1], float(args[2]))
	elif method == "dilate" and len(args) == 2:
		dilate(args[0], int(args[1]))
	elif method == "erode" and len(args) == 2:
		erode(args[0], int(args[1]))
	elif method == "opening" and len(args) == 2:
		opening(args[0], int(args[1]))
	elif method == "closing" and len(args) == 2:
		closing(args[0], int(args[1]))
	elif method == "tophatFilter" and len(args) == 3:
		tophatFilter(args[0], int(args[1]), args[2])
	elif method == "derivatives" and len(args) == 1:
		derivatives(args[0])
	elif method == "edgeCanny" and len(args) == 3:
		edgeCanny(float(args[0]), int(args[1]), int(args[2]))
	elif method == "edgeCanny" and len(args) == 4:
		edgeCanny(float(args[0]), int(args[1]), int(args[2]), args[3])
	elif method == "cornerHarris" and len(args) == 3:
		cornerHarris(float(args[0]), float(args[1]), int(args[2]))
	elif method == "cornerHarris" and len(args) == 4:
		cornerHarris(float(args[0]), float(args[1]), int(args[2]), args[3])
	else:
		print("> Método no reconocido ", method)
		print("> Parámetros pasados:", inputPath, outputName, histogram, debug, method, args)
		print(usage)
		sys.exit()
	plt.show()

if __name__ == "__main__":
	main(sys.argv[1:])
