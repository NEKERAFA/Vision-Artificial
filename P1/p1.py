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

plot = 0

def showHist(image, title=None):
	# Creo un nuevo plot
	global plot
	plot = plot+1
	plt.figure(plot)
	# Muestro el histograma de la imagen
	plt.hist(image.ravel(), range(0, 256))
	if not title is None:
		plt.title(title)

def showImg(image, title=None):
	# Creo un nuevo plot
	global plot
	plot = plot+1
	plt.figure(plot)
	# Muestro la imagen
	plt.imshow(image, cmap='gray', vmin = 0, vmax = 255)
	if not title is None:
		plt.title(title)

def Input(path, hist):
	original = misc.imread(path, True)
	showImg(original, 'Imagen de entrada')
	if hist:
		showHist(original, 'Histograma de entrada')
	return original

def Output(output, hist, path):
	if hist:
		showHist(output, 'Histograma de salida')
	showImg(output, 'Imagen de salida')
	if not path is None:
		misc.imsave(path, output)

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
			 '   --method = highBoost <A> <method> <parameter>')

	try:
		opts, args = getopt.getopt(argv, "m:i:o:h:", ["method=", "input=", "output=", "histogram="])
	except getopt.GetoptError:
		print("> Error al parsear")
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
	else:
		print("> Parámetros pasados:", method, inputPath, outputPath, hist, args)
		print(usage)
		sys.exit()

if __name__ == "__main__":
	main(sys.argv[1:])

#histEnhance()
#histAdapt()
#convolve()
#gauss()
#median()

#highboost()
