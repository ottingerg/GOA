#!/usr/bin/python3

import sys, getopt
import pickle
import psycopg2
from scipy.sparse import csr_matrix,lil_matrix
import numpy as np
import time

__author__ = "Georg Ottinger"
__copyright__ = "Copyright 2022, Georg Ottinger"
__credits__ = ["Georg Ottinger"]
__license__ = "GPL"
__version__ = "0.1.5"
__maintainer__ = "Georg Ottinger"
__email__ = "g.ottinger@gmx.at"
__status__ = "Prototype"
__date__ = "2021-03-18"

def usage():
	print("exportGSD --datebase DBNAME --infile FILE [--reset] [--boundary NAME] [--label LABEL]")
	sys.exit(2)

def main(argv):
	inputfile = ''
	database = ''
	label = ''
	boundary = ''
	droptables = False
	try:
		opts, args = getopt.getopt(argv,"hri:d:b:l:",["reset","infile=","database=","boundary=","label="])
	except getopt.GetoptError:
	  usage()
	for opt, arg in opts:
		if opt == '-h':
			usage()
		elif opt in ("-i", "--infile"):
			inputfile = arg
		elif opt in ("-d", "--database"):
			database = arg
		elif opt in ("-r", "--reset"):
			droptables = True
		elif opt in ("-b", "--boundary"):
			boundary = arg
		elif opt in ("-l", "--label"):
			label = arg

	if inputfile == '' or database == '':
		usage()


	with open(inputfile,'rb') as f:
		(verticies_ids,setlist,topsolution,residuals,clusters_r,stolen,clusters_s) = pickle.load(f)
		(topsolution_genom, topsolution_cost) = topsolution

		conn = psycopg2.connect("dbname="+database)

		with conn:
			cur = conn.cursor()

			if droptables:				
				try:
					cur.execute("DROP TABLE viz_cluster")
				except:
					conn.rollback()

				try:
					cur.execute("DROP TABLE territories")
				except:
					conn.rollback()

			conn.commit()
			
			try:
				cur.execute("CREATE TABLE viz_cluster (id text, line geometry, type text)")
			except:
				conn.rollback()

			conn.commit()

			i = 0
			for (id,cset,_) in setlist:
				if topsolution_genom[i] > 0:
					cur.execute("INSERT INTO viz_cluster VALUES (\'"+label+"\',ST_Transform(ST_SetSRID(ST_PointFromGeoHash(\'"+verticies_ids[id]+"\'),4326),32633),\'initial\')")
					for nid in cset:
						if id != nid:
							cur.execute("INSERT INTO viz_cluster VALUES (\'"+label+"\',ST_Transform(ST_SetSRID(ST_MakeLine(ST_PointFromGeoHash(\'"+verticies_ids[nid]+"\'),ST_PointFromGeoHash(\'"+verticies_ids[id]+"\')),4326),32633),\'knn\')")
				i = i + 1

			id = 0
			for (_,rid) in residuals:
				if rid >= 0:
					cur.execute("INSERT INTO viz_cluster VALUES (\'"+label+"\',ST_Transform(ST_SetSRID(ST_MakeLine(ST_PointFromGeoHash(\'"+verticies_ids[id]+"\'),ST_PointFromGeoHash(\'"+verticies_ids[rid]+"\')),4326),32633),\'residual\')")
				id = id + 1

			for s in stolen.keys():
				cur.execute("INSERT INTO viz_cluster VALUES (\'"+label+"\',ST_Transform(ST_SetSRID(ST_MakeLine(ST_PointFromGeoHash(\'"+verticies_ids[s]+"\'),ST_PointFromGeoHash(\'"+verticies_ids[stolen[s]]+"\')),4326),32633),\'stolen\')")
		

			try:
				cur.execute("DROP TABLE baseregions")
			except:
				conn.rollback()

			cur.execute("CREATE TABLE baseregions (vertex_id text, point geometry, cluster_id integer, voronoi geometry)")
			conn.commit()

			for (id, memb) in clusters_r:
				cur.execute("INSERT INTO viz_cluster VALUES (\'"+label+"\',ST_Transform(ST_SetSRID(ST_PointFromGeoHash(\'"+verticies_ids[id]+"\'),4326),32633),\'central\')")

				for (_,cid) in memb:

					if id != cid:
						cur.execute("INSERT INTO viz_cluster VALUES (\'"+label+"\',ST_Transform(ST_SetSRID(ST_MakeLine(ST_PointFromGeoHash(\'"+verticies_ids[cid]+"\'),ST_PointFromGeoHash(\'"+verticies_ids[id]+"\')),4326),32633),\'cluster_r\')")

			for (id, memb) in clusters_s:
				cur.execute("INSERT INTO viz_cluster VALUES (\'"+label+"\',ST_Transform(ST_SetSRID(ST_PointFromGeoHash(\'"+verticies_ids[id]+"\'),4326),32633),\'minsum\')")
			
				for (_,cid) in memb:
					cur.execute("INSERT INTO baseregions VALUES(\'"+verticies_ids[cid]+"\',(SELECT point from verticies WHERE id=\'"+verticies_ids[cid]+"\'),"+str(id)+")")

					if id != cid:
						cur.execute("INSERT INTO viz_cluster VALUES (\'"+label+"\',ST_Transform(ST_SetSRID(ST_MakeLine(ST_PointFromGeoHash(\'"+verticies_ids[cid]+"\'),ST_PointFromGeoHash(\'"+verticies_ids[id]+"\')),4326),32633),\'cluster_s\')")


			try:
				cur.execute("DROP TABLE voronoihelper")
			except:
				conn.rollback()

			cur.execute("CREATE TABLE voronoihelper (voronoi geometry)")
			conn.commit()
			cur.execute("insert into voronoihelper select (ST_Dump(ST_VoronoiPolygons(ST_Collect( ARRAY( SELECT point FROM baseregions))))).geom")
			cur.execute("CREATE INDEX voronoi_idx ON voronoihelper USING GIST(voronoi)")
			cur.execute("UPDATE baseregions SET voronoi=(SELECT voronoi FROM voronoihelper ORDER BY voronoi <-> point LIMIT 1)")
			
			try:
				cur.execute("CREATE TABLE territories (id text,cluster_id integer, centerpoint geometry, polygon geometry)")
			except:
				conn.rollback()
			conn.commit()

			cur.execute("INSERT INTO territories (select \'"+label+"\',cluster_id,NULL,ST_Union(b.voronoi) from baseregions as b group by cluster_id)")

			for (id, _) in clusters_s:
				cur.execute("UPDATE territories SET centerpoint=(SELECT point from verticies WHERE id=\'"+verticies_ids[id]+"\') WHERE cluster_id="+str(id))


			if boundary != '':
				cur.execute("UPDATE territories SET polygon = ST_Intersection(polygon,(SELECT way FROM planet_osm_polygon WHERE \"boundary\" = \'administrative\' AND \"name\" = \'"+boundary+"\')) WHERE \"id\"=\'"+label+"\'")


#TODO make functions to clarify tasks


if __name__ == "__main__":
   main(sys.argv[1:])