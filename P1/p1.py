#!/usr/bin/python3

'''
	Archivo con funciones de ejemplo
'''

import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import sys, getopt

import histograma
import filtrado
import operadores
import bordes

plot = 0

def showHist(image, title=None, vmin=0, vmax=255):
	# Creo un nuevo plot
	global plot
	plot = plot+1
	plt.figure(plot)
	# Muestro el histograma de la imagen
	plt.hist(image.ravel(), range(vmin, vmax + 1))
	if not title is None:
		plt.title(title)

def showImg(image, title=None, vmin=0, vmax=255):
	# Creo un nuevo plot
	global plot
	plot = plot+1
	plt.figure(plot)
	# Muestro la imagen
	plt.imshow(image, cmap = 'gray', vmin = vmin, vmax = vmax)
	if not title is None:
		plt.title(title)

def Input(path, hist):
	original = misc.imread(path, True)
	showImg(original, 'Imagen de entrada')
	if hist:
		showHist(original, 'Histograma de entrada')
	return original

def InputBinary(path, hist):
	original = misc.imread(path, True)
	for i in range(0, original.shape[0]):
		for j in range(0, original.shape[1]):
			if original[i][j] < 128:
				original[i][j] = 0
			else:
				original[i][j] = 1

	showImg(original, 'Imagen de entrada', 0, 1)
	if hist:
		showHist(original, 'Histograma de entrada', 0, 1)
	return original

def Output(output, hist, path):
	if not path is None:
		misc.imsave(path + '.png', output)
	if hist:
		showHist(output, 'Histograma de salida')
	showImg(output, 'Imagen de salida')

def OutputBinary(output, hist, path):
	if not path is None:
		misc.imsave(path + '.png', output)
	if hist:
		showHist(original, 'Histograma de entrada')
	showImg(output, 'Imagen de salida', 1, 0)

def OutputOperator(output, title, hist, histTitle, path):
	if not path is None:
		misc.imsave(path + '.png', output)
	if hist:
		showHist(output, histTitle, output.min(), output.max())
	showImg(output, title, output.min(), output.max())

def histEnhance(pathInput, center, win, hist, pathOutput):
	print("> Realce de contraste 'window-level' centrado en " + str(center) + " con ventana " + str(win))
	original = Input(pathInput, hist)
	output = histograma.histEnhance(original, center, win)
	print("> Hecho")
	Output(output, hist, pathOutput)
	plt.show()

def histAdapt(pathInput, minValue, maxValue, hist, pathOutput):
	print("> Compresion/estiramiento de histograma con valores [" + str(minValue) + ", " + str(maxValue) + "]")
	original = Input(pathInput, hist)
	output = histograma.histAdapt(original, minValue, maxValue)
	print("> Hecho")
	Output(output, hist, pathOutput)
	plt.show()

def convolve(pathInput, pathKernel, hist, pathOutput):
	print("> Prueba de convolución")
	original = Input(pathInput, hist)
	kernel = misc.imread(pathKernel, True)
	output = filtrado.convolve(original, kernel)
	print("> Hecho")
	Output(output, hist, pathOutput)
	plt.show()

def gaussian(pathInput, sigma, hist, pathOutput):
	print("> Filtro Gaussiano con sigma " + str(sigma))
	original = Input(pathInput, hist)
	output = filtrado.gaussianFilter2D(original, sigma)
	print("> Hecho")
	Output(output, hist, pathOutput)
	plt.show()

def median(pathInput, filterSize, hist, pathOutput):
	print("> Filtro de medianas con tamaño " + str(filterSize))
	original = Input(pathInput, hist)
	output = filtrado.medianFilter2D(original, filterSize)
	print("> Hecho")
	Output(output, hist, pathOutput)
	plt.show()

def highBoost(pathInput, A, method, parameter, hist, pathOutput):
	print("> High Boost con amplificación " + str(A) + " y método " + method + "(" + str(parameter) + ")")
	original = Input(pathInput, hist)
	output = filtrado.highBoost(original, A, method, parameter)
	print("> Hecho")
	output = histograma.histAdapt(output, 0, 255)
	Output(output, hist, pathOutput)
	plt.show()

def dilate(pathInput, ElType, size, hist, pathOutput):
	print("> Dilatación con EE " + ElType + " de tamaño " + str(size))
	original = InputBinary(pathInput, hist)
	output = operadores.dilate(original, ElType, size)
	print("> Hecho")
	OutputBinary(output, hist, pathOutput)
	plt.show()

def erode(pathInput, ElType, size, hist, pathOutput):
	print("> Erosion con EE " + ElType + " de tamaño " + str(size))
	original = InputBinary(pathInput, hist)
	output = operadores.erode(original, ElType, size)
	print("> Hecho")
	OutputBinary(output, hist, pathOutput)
	plt.show()

def opening(pathInput, ElType, size, hits, pathOutput):
	print("> Apertura con EE " + ElType + " de tamaño " + str(size))
	original = InputBinary(pathInput, hist)
	output = operadores.opening(original, ElType, size)
	print("> Hecho")
	OutputBinary(output, hist, pathOutput)
	plt.show()

def closing(pathInput, ElType, size, hist, pathOutput):
	print("> Cierre con EE " + ElType + " de tamaño " + str(size))
	original = InputBinary(pathInput, hist)
	output = operadores.closing(original, ElType, size)
	print("> Hecho")
	OutputBinary(output, hist, pathOutput)
	plt.show()

def tophatFilter(pathInput, ElType, size, mode, hist, pathOutput):
	print("> Filtro Top-Hat de modo " + mode + " con EE " + ElType + " de tamaño " + str(size))
	original = InputBinary(pathInput, hist)
	output = operadores.tophatFilter(original, ElType, size, mode)
	print("> Hecho")
	OutputBinary(output, hist, pathOutput)
	plt.show()

def derivatives(pathInput, operator, hist, pathOutput):
	print("> Derivadas primeras con " + operator)
	original = Input(pathInput, hist)
	[gx, gy] = bordes.derivatives(original, operator)
	print("> Hecho")
	OutputOperator(gx, 'Gradiente en x', hist, 'Histograma de Gx', pathOutput + '_Gx')
	OutputOperator(gy, 'Gradiente en y', hist, 'Histograma de Gy', pathOutput + '_Gy')
	plt.show()

def edgeCanny(pathInput, sigma, tlow, thigh, hist, pathOutput):
	print("> Operador Canny")
	original = Input(pathInput, False)
	magnitude, orientation = bordes.edgeCanny(original, sigma, tlow, thigh)
	print("> Hecho")
	OutputOperator(magnitude, 'Magnitud', hist, 'Histograma Magnitud', pathOutput + '_Magnitude')
	OutputOperator(orientation, 'Orientacion', hist, 'Histograma Orientación', pathOutput + '_Orientation')
	plt.show()

def main(argv):
	inputPath = 'example1.png'
	outputPath = None
	method = ''
	hist = False
	usage = ('./p1.py -m <method> -i <input> -o <output> -h <true | false> <args>\n\n'
	         'Methods and functions:\n'
			 '   --method = histEnhance <cenValue> <winSize>\n'
			 '   --method = histAdapt <minValue> <maxValue>\n'
			 '   --method = convolve <pathKernel>\n'
			 '   --method = gaussian <sigma>\n'
			 '   --method = median <filterSize>\n'
			 '   --method = highBoost <A> <method> <parameter>\n'
			 '   --method = dilate <ElType> <size>\n'
			 '   --method = erode <ElType> <size>\n'
			 '   --method = opening <ElType> <size>\n'
			 '   --method = closing <ElType> <size>\n'
			 '   --method = topHat <ElType> <size> <mode>\n'
			 '   --method = derivatives <operator>\n'
			 '   --method = canny <sigma> <tlow> <thigh>')

	try:
		opts, args = getopt.getopt(argv, "m:i:o:h:", ["method=", "input=", "output=", "histogram="])
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
		elif opt in ("-h", "--histograma"):
			if arg in ("true", "True"):
				hist = True
			elif arg in ("false", "False"):
				hist = False
			else:
				print('> Para el histograma usa true o false')
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
		dilate(inputPath, args[0], int(args[1]), outputPath)
	elif method == "erode" and len(args) == 2:
		erode(inputPath, args[0], int(args[1]), outputPath)
	elif method == "opening" and len(args) == 2:
		opening(inputPath, args[0], int(args[1]), outputPath)
	elif method == "closing" and len(args) == 2:
		closing(inputPath, args[0], int(args[1]), outputPath)
	elif method == "topHat" and len(args) == 3:
		tophatFilter(inputPath, args[0], int(args[1]), args[2], outputPath)
	elif method == "derivatives" and len(args) == 1:
		derivatives(inputPath, args[0], hist, outputPath)
	elif method == "canny" and len(args) == 3:
		edgeCanny(inputPath, float(args[0]), int(args[1]), int(args[2]), hist, outputPath)
	else:
		print("> Parámetros pasados:", method, inputPath, outputPath, hist, args)
		print(usage)
		sys.exit()

if __name__ == "__main__":
	main(sys.argv[1:])
