#!/usr/bin/python3

import sys, getopt
import pickle
import psycopg2
from scipy.sparse import csr_matrix,lil_matrix
import numpy as np
import graphknn
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

street_types = ['motorway','trunk','primary','secondary','tertiary','unclassified','residential','motorway_link','trunk_link','primary_link','secondary_link','tertiary_link','living_street','track','service']
dummyzero_distance = 1e-6

def usage():
	print("exportGSD --datebase DBNAME --outfile FILE [--smallbuildings]")
	sys.exit(2)

def deletetables(cur):
	try: 
		cur.execute("drop table streets")
	except:
		pass
	try: 
		cur.execute("drop table verticies")
	except:
		pass
	try: 
		cur.execute("drop table buildings")
	except:
		pass
	try: 
		cur.execute("drop table subgraphs")
	except:
		pass

def create_streets(cur):
	cur.execute("CREATE TABLE streets(id text, polychain geometry, tunnel text, PRIMARY KEY(id))")

	street_filter = ''

	for s in street_types:
		street_filter = street_filter + 'highway=\''+s+'\' OR '
	
	street_filter = street_filter + 'False'

	cur.execute("INSERT INTO streets SELECT concat(osm_id::text,'#',ST_GeoHash(ST_Transform(ST_StartPoint(way),4326)),'#',ST_GeoHash(ST_Transform(ST_EndPoint(way),4326))),way,CASE WHEN tunnel IS NULL THEN 'no' ELSE tunnel END FROM planet_osm_line WHERE "+street_filter)

	cur.execute("CREATE INDEX pc_idx ON streets USING GIST(polychain)")
	#print(street_filter)

def create_intersections(cur):
	cur.execute("CREATE TABLE verticies(id text, point geometry, weight float, PRIMARY KEY(id))")
	cur.execute("INSERT INTO verticies SELECT DISTINCT ST_GeoHash(ST_Transform((ST_DumpPoints(ST_Intersection(a.polychain, b.polychain))).geom,4326)),(ST_DumpPoints(ST_Intersection(a.polychain, b.polychain))).geom,0 FROM streets as a, streets as b WHERE a.id != b.id AND (ST_Touches(a.polychain,b.polychain) OR (ST_Intersects(a.polychain,b.polychain) AND a.tunnel = b.tunnel))")

def create_buildings(cur, smallbuildings):
	cur.execute("CREATE TABLE buildings(id text, centerpoint geometry, PRIMARY KEY(id))")

	if smallbuildings:
		cur.execute("INSERT INTO buildings SELECT ST_GeoHash(ST_Transform(ST_Centroid(way),4326)), ST_Centroid(way) FROM planet_osm_polygon WHERE \"building\" = 'yes' AND way_area >= 10")
	else:
		cur.execute("INSERT INTO buildings SELECT ST_GeoHash(ST_Transform(ST_Centroid(way),4326)), ST_Centroid(way) FROM planet_osm_polygon WHERE \"building\" = 'yes' AND \"addr:housenumber\" IS NOT NULL")

	cur.execute("CREATE INDEX cp_idx ON buildings USING GIST(centerpoint)")

	cur.execute("INSERT INTO verticies SELECT b.id,ST_ClosestPoint(s.polychain,b.centerpoint),1 FROM buildings AS b CROSS JOIN LATERAL (SELECT polychain FROM streets ORDER BY polychain <-> b.centerpoint LIMIT 1) AS s")

	cur.execute("CREATE INDEX v_idx ON verticies USING GIST(point)")

def create_subgraphs(cur):

	cur.execute("CREATE TABLE subgraphs(street_id text, vertex_id text, vertex_weight float, vertex_position float)")

	cur.execute("INSERT INTO subgraphs SELECT s.id, v.id, v.weight, (ST_LineLocatePoint(s.polychain,v.point) * ST_Length(s.polychain)) AS pos FROM streets AS s CROSS JOIN LATERAL (SELECT * FROM verticies WHERE ST_Intersects(point,ST_Buffer(s.polychain,0.001, 'endcap=round join=round'))) AS v ORDER BY s.id, pos")

def create_adjecencymatrix(cur):
	verticies_idx = dict()
	verticies_ids = list()
	verticies_weights = list()
	n_verticies = 0
	
	cur.execute("select distinct vertex_id,vertex_weight from subgraphs")
	while True:
		try:
			(vertex_id,vertex_weight) = cur.fetchone()
		except:
			break
		verticies_idx[vertex_id] = n_verticies
		verticies_ids.append(vertex_id)
		verticies_weights.append(vertex_weight)
		n_verticies += 1

	adjacency_matrix = lil_matrix((n_verticies,n_verticies))

	cur.execute("select * from subgraphs")
	(last_street_id, last_vertex_id,last_vertex_weight,last_vertex_position) = cur.fetchone()

	while True:
		try:
			(street_id, vertex_id,vertex_weight,vertex_position) = cur.fetchone()
		except:
			break

		if street_id == last_street_id:
			distance = vertex_position - last_vertex_position
			if distance == 0:
				distance = dummyzero_distance

			adjacency_matrix[verticies_idx[vertex_id],verticies_idx[last_vertex_id]] = distance
			adjacency_matrix[verticies_idx[last_vertex_id],verticies_idx[vertex_id]] = distance
		
		last_street_id = street_id
		last_vertex_id = vertex_id
		last_vertex_weight = vertex_weight
		last_vertex_position = vertex_position

	adjacency_matrix = adjacency_matrix.tocsr()

	return((verticies_ids,verticies_weights,adjacency_matrix))

def main(argv):
	outputfile = 'output.gsd'
	database = ''
	smallbuilings = False
	try:
		opts, args = getopt.getopt(argv,"hsd:o:",["smallbuildings","database=","outfile="])
	except getopt.GetoptError:
	  usage()
	for opt, arg in opts:
		if opt == '-h':
			usage()
		elif opt in ("-d", "--database"):
			database = arg
		elif opt in ("-o", "--outfile"):
			outputfile = arg
		elif opt == '-s':
			smallbuilings = True
	if database == '':
		usage()

	conn = psycopg2.connect("dbname="+database)

	with conn:
		cur = conn.cursor()
		deletetables(cur)
		conn.commit()
		starttime = time.time()
		create_streets(cur)
		create_intersections(cur)
		create_buildings(cur,smallbuilings)
		create_subgraphs(cur)
		with open(outputfile,'wb') as f:
			pickle.dump(create_adjecencymatrix(cur),f)
		endtime = time.time()

		print("GSD-Graph generation finished in "+str((endtime-starttime)*1000)+" ms")

if __name__ == "__main__":
   main(sys.argv[1:])