#!/usr/bin/python3

import sys, getopt
import psycopg2

__author__ = "Georg Ottinger"
__copyright__ = "Copyright 2021, Georg Ottinger"
__credits__ = ["Georg Ottinger"]
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = "Georg Ottinger"
__email__ = "g.ottinger@gmx.at"
__status__ = "Prototype"
__date__ = "2021-12-11"

def usage():
	print("boundOSM --datebase DBNAME --boundary NAME")
	sys.exit(2)

def main(argv):
	database = ''
	boundary = ''
	try:
		opts, args = getopt.getopt(argv,"hd:b:",["database=","boundary="])
	except getopt.GetoptError:
	  usage()
	for opt, arg in opts:
		if opt == '-h':
			usage()
		elif opt in ("-b", "--boundary"):
			boundary = arg
		elif opt in ("-d", "--database"):
			database = arg

	if boundary == '' or database == '':
		usage()


	conn = psycopg2.connect("dbname="+database)

	with conn:
		cur = conn.cursor()
		cur.execute("DELETE FROM planet_osm_line WHERE NOT ST_Intersects(way, (SELECT way FROM planet_osm_polygon WHERE \"boundary\" = 'administrative' AND \"name\" = '"+boundary+"'))")
		cur.execute("DELETE FROM planet_osm_polygon WHERE NOT ST_Intersects(way, (SELECT way FROM planet_osm_polygon WHERE \"boundary\" = 'administrative' AND \"name\" = '"+boundary+"'))")
		cur.execute("DELETE FROM planet_osm_point WHERE NOT ST_Intersects(way, (SELECT way FROM planet_osm_polygon WHERE \"boundary\" = 'administrative' AND \"name\" = '"+boundary+"'))")

if __name__ == "__main__":
   main(sys.argv[1:])