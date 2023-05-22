import numpy as np
import os

GRAPH_FOLDER = './graph/'

def getFullFilePath(filename):
	return os.path.join(GRAPH_FOLDER, filename)

def saveGraph(plt, filename):
	plt.savefig(getFullFilePath(filename), bbox_inches='tight')
	plt.close()