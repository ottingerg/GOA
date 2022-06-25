#!/usr/bin/python3

import sys, getopt
import pickle
import numpy as np
import time
from random import random, sample, shuffle
from operator import itemgetter
import signal

import graphknn

__author__ = "Georg Ottinger"
__copyright__ = "Copyright 2022, Georg Ottinger"
__credits__ = ["Georg Ottinger"]
__license__ = "GPL"
__version__ = "0.3.7"
__maintainer__ = "Georg Ottinger"
__email__ = "g.ottinger@gmx.at"
__status__ = "Prototype"
__date__ = "2022-03-18"

anonymity = 5
search_nn = 30
cost_notconnected = 1e6

stopsignal = False

def signal_handler(sig, frame):
	global stopsignal
	print('\nCaught CTRL+C. finishing ...')
	stopsignal = True


def genetic_getcost(indvidiual):
	(_,cost) = indvidiual
	return(cost)

def genetic_fixsolution_calccost(individual,setlist,n_geoobjects,residualsonly=False):
	
	fixed = np.array(np.zeros(len(setlist)),dtype=int)

	i = 0
	shuffeled_order = list(range(len(setlist)))
	shuffle(shuffeled_order) #this needs documentation

	ind_set = set()
	
	cost = 0
	n = 0

	#merge individuals
	for i in shuffeled_order:
		if individual[i] > 0:
			(_,cset,setcost) = setlist[i]
			if ind_set.isdisjoint(cset):
				fixed[i] = individual[i]
				ind_set = ind_set.union(cset)
				if not residualsonly:
					cost = cost + setcost
				n = n + 1

	#try if additional sets fit - this step is important, cause otherwise the merge 
	#operation will give degenerated results, this would destroy the crossover aspect 
	#of the algorithm, because it would degenerate to a propalistic hill-climb
	#based on mutation and eltism.
	shuffle(shuffeled_order) #this needs documentation
	for i in shuffeled_order:
		if fixed[i] == 0:
			(_,cset,setcost) = setlist[i]
			if ind_set.isdisjoint(cset):
				fixed[i] = 1
				ind_set = ind_set.union(cset)
				if not residualsonly:
					cost = cost + setcost
				n = n + 1
					   

	cost = cost + (n_geoobjects - n * anonymity) * cost_notconnected
	

	return((fixed,cost))

def genetic_crossover(ind1, ind2, singlepoint=False):
	crossed = np.array(np.zeros(len(ind1)),dtype=int)

	if singlepoint:
		crossoverpoint = random()*len(ind1)
		for i in range(len(ind1)):
			if i<crossoverpoint:
				crossed[i] = ind1[i]
			else:
				crossed[i] = ind2[i]  
	else:
		for i in range(len(ind1)):
			if i%2:
				crossed[i] = ind1[i]
			else:
				crossed[i] = ind2[i]

		
	return(crossed)

def genetic_mutate(ind, rate):
	mutated = ind

	for i in range(len(mutated)):
		if random() < rate:
			mutated[i] = not bool(mutated[i])

	return(mutated)

def genetic_getmincost(population):
	(_,cost) = population[0]

	for (_,newcost) in population[1:]:
		if newcost < cost:
			cost = newcost

	return(cost)

def visualize_intemrediate_results(round,solution,verticies_ids,knn,setlist,outputfilebase):
	

	residuals = [(-1,-1) for n in range(len(verticies_ids))]

	clusters_r = get_clusters(setlist,solution,residuals,knn)

	outputfile = outputfilebase + "."+str(round)+"_"+str(genetic_getcost(solution))
	
	with open(outputfile,'wb') as g:
		pickle.dump((verticies_ids,setlist,solution,list(),clusters_r,dict(),list()),g)

	


def genetic_find_sets(setlist,n_geoobjects,knn,verticies_ids,n_rounds=2000,n_population=10,elitism=1,mutation_rate=5,singlepoint=False,visualize=False,outputfile=''):
	global stopsignal
	start_time = time.time()

	print("[GA Select] Parameters: Population="+str(n_population)+", Rounds="+str(n_rounds)+", Elitism="+str(elitism)+", Mutationrate="+str(mutation_rate))

	if n_rounds <= 10:
		newlinecount = 1
	elif n_rounds <= 100:
		newlinecount = 10
	else:
		newlinecount = 100

	population = list()

	for j in range(n_population):
		individual = np.array(np.ones(len(setlist)),dtype=int)
  
		population.append(genetic_fixsolution_calccost(individual,setlist,n_geoobjects))

	elite_population = sorted(population,key=itemgetter(1),reverse=False)


	for round in range(n_rounds):

		if visualize:
			visualize_intemrediate_results(round,sorted(population,key=itemgetter(1),reverse=False)[0],verticies_ids,knn,setlist,outputfile)

		if ((round) % newlinecount) == 0:
			print("\r[GA Select] Round "+str(round)+" Minimum Cost: "+str(genetic_getmincost(population)))
		else:
			print("\r[GA Select] Round "+str(round)+" Minimum Cost: "+str(genetic_getmincost(population)),end="")

		if stopsignal:
			break

		potential_mothers = population.copy()
		shuffle(potential_mothers)
		potential_fathers = population
		shuffle(potential_fathers)

		new_population = list()

		for i in range(len(population)-elitism):
			#mother tournament
			if genetic_getcost(potential_mothers[i]) < genetic_getcost(potential_mothers[(i+1)%n_population]):
				mother = potential_mothers[i]
			else:
				mother = potential_mothers[(i+1)%n_population]

			#father tournament
			if genetic_getcost(potential_fathers[i]) < genetic_getcost(potential_fathers[(i+1)%n_population]):
				father = potential_fathers[i]
			else:
				father = potential_fathers[(i+1)%n_population]

			child = genetic_mutate(genetic_crossover(mother[0],father[0],singlepoint=singlepoint),mutation_rate/len(setlist))

			new_population.append(genetic_fixsolution_calccost(child,setlist,n_geoobjects))

		elite_population = sorted(population,key=itemgetter(1),reverse=False)
		for i in range(elitism):
			new_population.append(elite_population[i])

		population = new_population


	if stopsignal:
		print()	


	if visualize:
		visualize_intemrediate_results(round+1,sorted(population,key=itemgetter(1),reverse=False)[0],verticies_ids,knn,setlist,outputfile)

	print("\r[GA Select] Round "+str(round+1)+" Minimum Cost: "+str(genetic_getmincost(population)))

	count_s = 0
	for s in elite_population[0][0]:
		if s:
			count_s = count_s +1

	print("[GA Select] "+str(count_s)+" sets selected.")
	print("[GA Select] took "+format(time.time()-start_time,"3.3f") +" secs.")

	return(elite_population[0])

def unique_sets(knn, verticies_weights,minsum = False):
	start_time = time.time()
	sets = dict()
	geoobject_ids = set()
	
	setsizes = [0 for n in range(anonymity+1)]

	i = 0
	for k in knn:
		k = k[:anonymity]
		if verticies_weights[i] > 0:
			setsizes[len(k)] = setsizes[len(k)] + 1

		if verticies_weights[i] > 0 and len(k) == anonymity:
			cost = 0
			s = set()
			for (c,m) in k:
				cost = cost + c
				s.add(m)
				geoobject_ids.add(m)
			excentricity = k[-1][0]  # distance of last next neighbour
			if frozenset(s) not in sets:
				sets[frozenset(s)] = (i,cost,excentricity)
			else:
				(_,s_cost,s_exzentricity) = sets[frozenset(s)]
				if minsum:
					if cost < s_cost:
						sets[frozenset(s)] = (i,cost,excentricity)
				else:
					if excentricity < s_exzentricity:
						sets[frozenset(s)] = (i,cost,excentricity)
			
		i = i + 1
	setlist = list()

	for key in sets.keys():
		(v,cost,_) = sets[key]
		setlist.append((v,key,cost))

	for i in range(anonymity+1):
		if setsizes[i]:
			print("[Select Sets] Number of "+str(i)+"-sets: "+str(setsizes[i]))

	print("[Select Sets] Number of unique "+str(anonymity)+"-sets: "+str(len(setlist)))
	print("[Select Sets] took "+format(time.time()-start_time,"3.3f") +" secs.")

	return (setlist,geoobject_ids)

		
def assign_residuals(knn,setlist,topsolution,verticies_weights,geoobject_ids):
	start_time = time.time()
	n_verticies = len(verticies_weights)

	residuals = [0 for n in range(n_verticies)]
	residuals_assigned = [(-1,-1) for n in range(n_verticies)]

	i = 0
	for w in verticies_weights:
		if w > 0 and i in geoobject_ids:
			residuals[i] = w
		i = i + 1

	(ind,_) = topsolution

	centers = set()
	i = 0	
	for (v,cset,_) in setlist:
		if ind[i]: 
			centers.add(v)
			for c in cset:
				residuals[c] = 0
		i = i + 1

	ra = 0
	for r in range(len(residuals)):
		if residuals[r]:
			for (d,n) in knn[r][1:]:
				if n in centers:
					residuals_assigned[r] = (d,n)
					residuals[r] = 0
					ra = ra + 1
					break			
		

	print("[Residuals] Assigned "+str(ra)+" residuals, "+str(residuals.count(1)) + " remaining.")
	print("[Residuals] took "+format(time.time()-start_time,"3.3f") +" secs.")

	return(residuals_assigned)

def get_central_vertex(vset, n_verticies, adjacency_matrix, minsum = False):

	labeled = np.zeros(n_verticies,dtype=bool)

	for v in vset:
		labeled[v] = True

	knn = graphknn.algorithm1earlyexit(adjacency_matrix,labeled,len(vset))

	if minsum:
		max_cost = cost_notconnected
		for v in vset:
			cost= 0
			for (d,_) in knn[v]:
				cost = cost + d
			if cost < max_cost:
				max_cost = cost
				center_v = v

	else: #calculate central vertex with exzentricity	
		oneelement = sample(vset, 1) 
		vset2 = vset.difference(set(oneelement))

		center_v = oneelement[0]
		excentricity = knn[center_v][-1][0]

		for v in vset2:
			if knn[v][-1][0] < excentricity:
				center_v = v
				excentricity = knn[v][-1][0]

	return((center_v,knn[center_v]))		

def get_clusters(setlist,topsolution,residuals,knn):

	clusterlist = list(list() for i in range(len(residuals)))

	i = 0
	for (v,_,_) in setlist:
		if topsolution[0][i] > 0:
			clusterlist[v] = knn[v][:anonymity]
		i = i + 1

	for r in range(len(residuals)):
		(d,v_c) = residuals[r]
		if v_c >= 0:
			clusterlist[v_c].append((d,r))


	cluster = list()
	i = 0
	for s in clusterlist:
		if s != list():
			cluster.append((i,s))
		i = i +1

	return(cluster)
	
	

def recalc_centers(clusters,adjacency_matrix,minsum = False):
	start_time = time.time()
	n_verticies = adjacency_matrix.shape[0]

	choosensets = list(set() for i in range(n_verticies))

	for (c_v,vset) in clusters:
		for (_,v) in vset:
			choosensets[c_v].add(v)
	
	newcluster = list()
	i = 0
	moved_centers = 0
	for s in choosensets:
		if s != set():
			(center,tgset) = get_central_vertex(s,n_verticies,adjacency_matrix,minsum=minsum)
			if center != i:
				moved_centers = moved_centers + 1
			
			newcluster.append((center,tgset))
		i = i + 1
	
	if minsum:
		print("[Recalc Centers] moved "+str(moved_centers)+" centers. (minsum)")
	else:
		print("[Recalc Centers] moved "+str(moved_centers)+" centers. (central vertex)")

	print("[Recalc Centers] took "+format(time.time()-start_time,"3.3f") +" secs.")

	return(newcluster)

def steal_nodes_from_a_to_b(a,b,maybeb,assigned_center):


	stolen = dict()

	adict = dict()
	for (d,v) in a[1:]:
		adict[v] = d

	bdict = dict()
	for (d,v) in b[1:]:
		bdict[v] = d

	maybebdict = dict()
	for (d,v) in maybeb[1:]:
		maybebdict[v] = d

	adictcopy = dict(sorted(adict.items(), key=lambda item: item[1],reverse=True))

	for v in adictcopy.keys():
		if len(adict) <= (anonymity-1):
			break 
		if v in maybebdict.keys():
			if adict[v] > maybebdict[v]:
				bdict[v] = maybebdict[v]
				assigned_center[v] = b[0][1]
				del adict[v]
				stolen[v] = b[0][1]

				
	asorted = dict(sorted(adict.items(), key=lambda item: item[1]))
	bsorted = dict(sorted(bdict.items(), key=lambda item: item[1]))

	alist = list()
	alist.append(a[0])
	blist = list()
	blist.append(b[0])

	for key in asorted.keys():
		alist.append((asorted[key],key))

	for key in bsorted.keys():
		blist.append((bsorted[key],key))

	return(alist,blist,assigned_center,stolen)

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def steal_nodes(clusterslist, knn):
	start_time = time.time()

	clusters = dict()
	assigned_center = dict()
	clusters_centers = set()
	non_centers = set()

	maxcluster_size = 0
	for (center_v,memb) in clusterslist:
		clusters[center_v] = memb
		if len(memb) > maxcluster_size:
			maxcluster_size = len(memb)
		for (_,v) in memb[1:]:
			non_centers.add(v)
			assigned_center[v] = center_v

		clusters_centers.add(center_v)


	allstolen = dict()

	for v in non_centers:
		oldcenter = assigned_center[v]
		for (_,newcenter) in knn[v][1:]:
			try: 
				if newcenter == assigned_center[v]:
					break #already assigned to nearest center
				elif newcenter in clusters_centers:
					(clusters[oldcenter],clusters[newcenter],assigned_center,stolen) = steal_nodes_from_a_to_b(clusters[oldcenter],clusters[newcenter],knn[newcenter],assigned_center)							
					allstolen = merge_two_dicts(allstolen,stolen)
					break
			except:
				pass


	clusterlist = list()
	for key in clusters.keys():
		clusterlist.append((key,clusters[key]))
	
	print("[Steal Nodes] "+str(len(allstolen))+" nodes have been stolen.")
	print("[Steal Nodes] took "+format(time.time()-start_time,"3.3f") +" secs.")

	return(clusterlist,allstolen)


def usage():
	print("runGOA --infile INFILE --outfile OUTFILE [--rounds ROUNDS] [--population NUMPOP] [--mutationrate RATE] [--elitism NUMELITS] [--nostealnodes] [--visualize]")
	sys.exit(2)

def main(argv):
	global stopsignal
	inputfile = ''
	outputfile = ''
	rounds = 100
	population = 100
	mutationrate = 1
	tries = 1
	elitism = 1
	stealnodes = True
	singlepoint = False
	visualize = False

	try:
		opts, args = getopt.getopt(argv,"hnsvi:o:r:p:m:t:e:",["singlepoint","visualize","infile=","outfile=","rounds=","population=","mutationrate=","tries=","elitism="])
	except getopt.GetoptError:
		usage()
	for opt, arg in opts:
		if opt == '-h':
			usage()
		elif opt in ("-i", "--infile"):
			inputfile = arg
		elif opt in ("-o", "--outfile"):
			outputfile = arg
		elif opt in ("-r", "--rounds"):
			rounds = int(arg)
		elif opt in ("-p", "--population"):
			population = int(arg)
		elif opt in ("-m", "--mutatiorate"):
			mutationrate = int(arg)
		elif opt in ("-e", "--elitism"):
			elitism = int(arg)
		elif opt in ("-t", "--tries"):
			tries = int(arg)
		elif opt in ("-n", "--nostealnodes"):
			stealnodes = False
		elif opt in ("-s", "--singlepoint"):
			singlepoint = True
		elif opt in ("-v", "--visualize"):
			visualize = True

	start_time = time.time()

	if inputfile == '' or outputfile == '':
		usage()

	signal.signal(signal.SIGINT, signal_handler)

	with open(inputfile,'rb') as f:
		(verticies_ids,verticies_weights,adjacency_matrix) = pickle.load(f)

		print("[geodesic kNN] run with search range of "+str(search_nn)+" nearest neighbours.")
		start_time = time.time()
		knn = graphknn.algorithm1earlyexit(adjacency_matrix,np.array(verticies_weights,dtype=bool),search_nn)
		print("[geodesic kNN] took "+format(time.time()-start_time,"3.3f") +" secs.")
		
		(setlist,geoobjects_ids) = unique_sets(knn, verticies_weights)
		n_geoobjects = len(geoobjects_ids)

		topsolution=(list(),cost_notconnected * n_geoobjects)

		for i in range(tries):
			(genom,cost)=genetic_find_sets(setlist,n_geoobjects,knn, verticies_ids,n_rounds=rounds,n_population=population, mutation_rate=mutationrate,elitism=elitism,singlepoint=singlepoint,visualize=visualize,outputfile=outputfile)
			if cost < genetic_getcost(topsolution):
				topsolution = (genom,cost)
			if stopsignal:
				break
		
		residuals=assign_residuals(knn,setlist,topsolution,verticies_weights,geoobjects_ids)
		
		clusters_r = get_clusters(setlist,topsolution,residuals,knn)

		if stealnodes:
			clusters_r = recalc_centers(clusters_r, adjacency_matrix)
			(clusters_s,stolen) = steal_nodes(clusters_r,knn)
		else:
			clusters_s = clusters_r
			stolen=dict()
			
		clusters_s = recalc_centers(clusters_s, adjacency_matrix, minsum=True)

		with open(outputfile,'wb') as g:
			pickle.dump((verticies_ids,setlist,topsolution,residuals,clusters_r,stolen, clusters_s),g)

	print("[GOA total time] took "+format(time.time()-start_time,"3.3f") +" secs.")

if __name__ == "__main__":
   main(sys.argv[1:])