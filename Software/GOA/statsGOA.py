#!/usr/bin/python3

import sys, getopt
import pickle
import time
from statistics import mean,stdev
import math
from scipy.stats import skew,scoreatpercentile

__author__ = "Georg Ottinger"
__copyright__ = "Copyright 2022, Georg Ottinger"
__credits__ = ["Georg Ottinger"]
__license__ = "GPL"
__version__ = "0.1.5"
__maintainer__ = "Georg Ottinger"
__email__ = "g.ottinger@gmx.at"
__status__ = "Prototype"
__date__ = "2021-01-06"

def usage():
	print("statsGOA --infile GOAFILE")
	sys.exit(2)

max_supported_clustersize = 20

def cluster_stats(clusters):


	n_bins = 10

	cluster_histogram = list(0 for i in range(max_supported_clustersize))
	cluster_sizes = list()
	ccosts = cluster_histogram[:]
	n_objects = 0
	costs = list()
	for (id,memb) in clusters:
		size = len(memb)

		n_objects = n_objects + size

		cluster_sizes.append(size)
		cluster_histogram[size] = cluster_histogram[size] + 1 

		for (d,_) in memb:
			costs.append(d)
			ccosts[size] = ccosts[size] + d


	print("Number of objects           : "+format(n_objects,'6d'))
	print("Number of clusters          : "+format(len(clusters),'6d'))

	print("=== Cluster Size Stats ===")
	print("Minimum cluster size        : "+format(min(cluster_sizes),'2d'))
	print("Maximum cluster size        : "+format(max(cluster_sizes),'2d'))
	print("Average cluster size        : "+format(mean(cluster_sizes),'4.2f'))		
	print("Cluster size stddev         : "+format(stdev(cluster_sizes),"4.2f"))
	print("Cluster size skewness       : "+format(skew(cluster_sizes),"4.2f"))
	print("Cluster size 50% percentile : "+format(scoreatpercentile(cluster_sizes,50),"4.2f"))
	print("Cluster size 95% percentile : "+format(scoreatpercentile(cluster_sizes,95),"4.2f"))
	print("Cluster size 99% percentile : "+format(scoreatpercentile(cluster_sizes,99),"4.2f"))

	print("   --- Cluster Size Histogram ---")

	for i in range(max(cluster_sizes)):
		if cluster_histogram[i+1] > 0:
			print("Number of "+format(i+1,'2d')+"-cluster        : "+format(cluster_histogram[i+1],'6d')+"     Average vertex cost [m]: "+format((ccosts[i+1]/cluster_histogram[i+1]/(i+1)),'.2f'))

	print("=== Vertex Cost Stats ===")
	print("Average vertex cost [m]     : "+format(mean(costs),"4.2f"))
	print("Minimum vertex cost [m]     : "+format(min(costs),"4.2f"))
	print("Maximum vertex cost [m]     : "+format(max(costs),"4.2f"))
	print("Vertex cost stddev[m]       : "+format(stdev(costs),"4.2f"))
	print("Vertex cost skewness        : "+format(skew(costs),"4.2f"))
	print("Vertex cost 50% percentile  : "+format(scoreatpercentile(costs,50),"4.2f"))
	print("Vertex cost 95% percentile  : "+format(scoreatpercentile(costs,95),"4.2f"))
	print("Vertex cost 99% percentile  : "+format(scoreatpercentile(costs,99),"4.2f"))
	
	print("   --- Vertex Cost Histogram ---")

	bin_factor = math.ceil(max(costs)/100)*100/n_bins
	
	bins = list(0 for i in range(n_bins))

	for d in costs:
		i = math.floor(d/bin_factor)
		bins[i] = bins[i] + 1

	for i in range(n_bins):
		print("Number of Vertex Costs from ["+str(i*bin_factor)+","+str((i+1)*bin_factor)+"[ [m]: \t"+str(bins[i]))

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
		(verticies_ids,setlist,topsolution,residuals,clusters_r,stolen,clusters_s) = pickle.load(f)
		(topsolution_genom, topsolution_cost) = topsolution
		cluster_stats(clusters_s)

if __name__ == "__main__":
   main(sys.argv[1:])