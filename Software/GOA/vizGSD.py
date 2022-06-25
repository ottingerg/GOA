#!/usr/bin/python3

import sys, getopt
import pickle
import psycopg2
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
__date__ = "2021-12-07"

def usage():
	print("vizGSD --datebase DBNAME --infile FILE")
	sys.exit(2)

def main(argv):
	inputfile = ''
	database = ''
	try:
		opts, args = getopt.getopt(argv,"hd:i:",["database=","infile="])
	except getopt.GetoptError:
	  usage()
	for opt, arg in opts:
		if opt == '-h':
			usage()
		elif opt in ("-d", "--database"):
			database = arg
		elif opt in ("-i", "--infile"):
			inputfile = arg
	if database == '':
		usage()
	if inputfile == '':
		usage()

	conn = psycopg2.connect("dbname="+database)

	with conn:
		cur = conn.cursor()
		try: 
			cur.execute("drop table edges")
		except:
			pass
		conn.commit()

		cur.execute("CREATE TABLE edges(a_id text, b_id text, distance float, line geometry)")

		with open(inputfile,'rb') as f:
			(verticies_ids,verticies_weights,adjacency_matrix) = pickle.load(f)
			n_verticies = len(verticies_weights)

			i = 0
			for x in range(0,n_verticies):
				print("\rprogress "+format(i/n_verticies*100,'.2f')+" %",end="")
				for y in range(x,n_verticies):
					distance = adjacency_matrix[x,y]
					if distance != 0:
						#print(x,distance)
						cur.execute("INSERT INTO edges VALUES(\'"+verticies_ids[x]+"\',\'"+verticies_ids[y]+"\',"+str(distance)+",ST_Transform(ST_SetSRID(ST_MakeLine(ST_PointFromGeoHash(\'"+verticies_ids[x]+"\'),ST_PointFromGeoHash(\'"+verticies_ids[y]+"\')),4326),32633))")
				i = i + 1
			print("\rprogress "+format(i/n_verticies*100,'.2f')+" %")

	
if __name__ == "__main__":
   main(sys.argv[1:])