#!/usr/bin/python3

import sys, getopt
import pickle
from numpy.core.fromnumeric import nonzero
from scipy.sparse import csr_matrix,lil_matrix
import numpy as np
import time

__author__ = "Georg Ottinger"
__copyright__ = "Copyright 2021, Georg Ottinger"
__credits__ = ["Georg Ottinger"]
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = "Georg Ottinger"
__email__ = "g.ottinger@gmx.at"
__status__ = "Prototype"
__date__ = "2021-12-10"

def usage():
	print("statsGSD --infile GSDFILE")
	sys.exit(2)

def crossing_degree(verticies_weights, adjacency_matrix,degree=2):
	n = len(verticies_weights)
	rc_sum = 0
	rc_count = 0
	v = 0
	for w in verticies_weights:
		if w > 0:
			continue
		#print(adjacency_matrix[v].nonzero()[1])
		deg_v = len(adjacency_matrix[v].nonzero()[1])
		v = v + 1
		if deg_v > degree:
			rc_sum = rc_sum + deg_v
			rc_count = rc_count + 1

	return (rc_sum/rc_count)

def get_intersections(verticies_weights,adjacency_matrix):

	i = 0 
	count = 0
	for w in verticies_weights:
		if w == 0:
			if len(adjacency_matrix[i].nonzero()[0]) > 2:
				count = count + 1
		i = i + 1 

	return (count)

def main(argv):
	inputfile = ''
	try:
		opts, args = getopt.getopt(argv,"hi:",["infile="])
	except getopt.GetoptError:
	  usage()
	for opt, arg in opts:
		if opt == '-h':
			usage()
		elif opt in ("-i", "--infile"):
			inputfile = arg
	if inputfile == '':
		usage()

	with open(inputfile,'rb') as f:
		(verticies_ids,verticies_weights,adjacency_matrix) = pickle.load(f)
		edges = int(len(adjacency_matrix.nonzero()[1])/2)
		n_verticies = len(verticies_weights)
		print("Number of verticies                : "+str(n_verticies))
		print("Number of edges                    : "+str(edges))
		print("Graph densitiy                     : "+str(edges*2/(n_verticies * (n_verticies -1))))
		print("Sum of edge distances              : "+str(adjacency_matrix.sum()/2))
		print("Number of intersections            : "+str(verticies_weights.count(0)))
		print("Number of real intersects (deg>2)  : "+str(get_intersections(verticies_weights,adjacency_matrix)))
		print("Number of geoobjects               : "+str(verticies_weights.count(1)))
		print("Average degree of verticies        : "+str(crossing_degree(verticies_weights, adjacency_matrix,degree=0)))
		print("Average degree of real intersection: "+str(crossing_degree(verticies_weights, adjacency_matrix,degree=2)))
		

if __name__ == "__main__":
   main(sys.argv[1:])