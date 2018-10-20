from pyspark                  import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy                    import array

print("rdd script")

def closestPoint(p, dict):
      bestIndex = 0
      particle = 0
      closest = float("+inf")
      key_iterator = 0
      final_new_array = [[]] * no_of_particles
      for key,value in dict.items():
          v1= value
          for i in range(len(v1)):
               tempDist = numpy.sqrt(numpy.sum(numpy.subtract(p, v1[i][1]) ** 2))
               if tempDist < closest:
                  closest = tempDist
                  bestIndex = v1[i][0]
                  particle = key
          key_iterator = key_iterator + 1
          final_new_array[key] = ((particle,bestIndex), [p, closest])
          closest = float("+inf")
      return final_new_array

def thedis(p, lis):
      bestIndex = 0
      particle = 0
      closest = float("+inf")
      for i in range(len(lis)):
          tempDist = numpy.sqrt(numpy.sum(numpy.subtract(p, lis[i][1]) ** 2))
          if tempDist < closest:
             closest = tempDist
             bestIndex = lis[i][0]
      final_array = [bestIndex, closest]
      return final_array

def summation_by_reduce(x, y):
	if x is not None or y is not None:
		if x is None:
			return y
		elif y is None:
			return x
		else:
			return [(x[0][0]+y[0][0], x[0][1]+y[0][1]),x[1]+y[1]]

def add_tuples_tolist_closest(x, y):
	if len(x) == 2:
		val = x[1]
		if type(val) != list:
			return (x[0],y[0])
			#return [x[0]] + [y[0]]
	return x + (y[0],)
	#return x + [y[0]]

def cen_cal(p,q):
    new_temp_arrlis = []
    for i in range(len(p)):
         for j in range(len(q)):
             if(p[i] == q[j][0]):
				return [q[j][0],(list(numpy.divide(p[1][0],q[j][1])),p[1][1])]

def cen_cal_2(p,q):
    new_temp_arrlis = []
    for i in range(len(p)):
         for j in range(len(q)):
             if(p[i] == q[j][0]):
                return (q[j][0],[list(numpy.divide(p[1][0],q[j][1])),p[1][1]])

def dummyfun(p,q):
	fi_arr = []
	if (p[1][4] < p[1][1]):
		variation = numpy.subtract(p[1][3],p[1][0])
	else:
		variation = numpy.subtract(p[1][0],p[1][3])
	for i in range(len(q)):
		if (q[i][0] == p[0]):
			qp_charg = q[i][1]
		elif (q[i][0] == p[1][2]):
			qh_charge = q[i][1]
	charge_mul = qp_charg*qh_charge
	dd = map(lambda x:x*charge_mul,variation)
	pairedForce = dd/p[1][5]
	fi_arr.append((p[0],list(pairedForce)))
	return fi_arr

def chargefun(p,q,r):
    charge = []
    for i in range(len(p)):
           numerator = p[i][1][1] - q[1]
           temp_fin_val = numpy.exp((p[i][0][0]+1)*(numerator/r))
           charge.append((p[i][0],temp_fin_val))
    return charge

def add_tuples_tolist(x, y):
	if type(x) == tuple:
		return [x, y]
	else:	
		if type(y) == list: 
			return x+y
		else:
			return x+[y]

def limitsfun(x):
	dist_arr = [] 
	k = 0
	cent = x[1][0]
	for i in range(len(x[1])):
		gg = x[1][i]
		if i != 0:
			if type(gg) == list:
				dist = numpy.sqrt(numpy.sum(numpy.subtract(cent, gg) ** 2))
				dist_arr.append((x[1][i],dist))
	dist_arr.sort(key=itemgetter(1))
	dimen = len(dist_arr)
	return ((x[0]),(cent,dist_arr[0],dist_arr[dimen-1]))

def movement_fun(p):
	lam = numpy.random.random(1)[0]
	final_move = []
	if p[1][0] != numpy.nan and p[1][1] != numpy.nan:
		if (p[1][0] + p[1][1]) <= 0:
			move = lam * (p[1][0] + p[1][1]) 
			sub = list(numpy.subtract(p[1][2],p[1][4][0]))
			final_move = list(numpy.multiply(move,sub))
		else:
			move = lam * (p[1][0] + p[1][1])
			sub = list(numpy.subtract(p[1][3][0],p[1][2]))
			final_move = list(numpy.multiply(move,sub))
		return (p[0],list(numpy.sum([p[1][2],final_move],axis=0)))
	else:
		return 0

def dict_to_rdd(dict):
	dict_to_arr = []
	for key,value in dict.items():
		v1= value
		for i in range(len(v1)):
			val = (key,v1[i][0])
			lis = v1[i][1]
			dict_to_arr.append((val,lis))
	return dict_to_arr

import numpy
from operator import itemgetter, attrgetter
from itertools import groupby
sc   = SparkContext()
data = sc.textFile("bezdekIris.csv")
data2 = data.map(lambda x:x.split(','))
data3= data2.map(lambda x:(x[0],x[1]))
data4= data3.filter(lambda x:x != ('Sepal length','Sepal width'))
data5 = data4.map(lambda lines:[float(x) for x in lines])
no_of_particles = 4
final_temp = [[[]]] * no_of_particles 
index_list = [] * no_of_particles
new_arr = [[]] * no_of_particles 
for i in range(no_of_particles):
    index_list.append(i)

for i in range(no_of_particles):
    new_arr[i] = data5.takeSample(False,no_of_particles,i+i)
    final_temp[i] = map(list,list(zip(index_list,new_arr[i])))

dict1 = dict(zip(index_list,final_temp))
for i in range(10):
		closest = data5.flatMap(lambda p: (closestPoint(p, dict1)))
		points_grp = closest.reduceByKey(lambda x,y:add_tuples_tolist_closest(x,y))		
		reduced = closest.reduceByKey(lambda x,y:summation_by_reduce(x, y))
		total_counts = [x for x in closest.map(lambda t:(t[0],1)).reduceByKey(lambda x,y:x+y).toLocalIterator()]
		partcl_centroid = reduced.map(lambda p:cen_cal(p,total_counts))
		grp_particles = partcl_centroid.groupBy(lambda p:p[0][0]).map(lambda p:p[1])
		xbest_finder = grp_particles.map(lambda x: [a[1][1] for a in x])
		xbest_ini = [x for x in xbest_finder.map(lambda x: reduce(lambda a,b: numpy.sum([a, b], axis=0),x)).toLocalIterator()]
		particles_ini = [x for x in partcl_centroid.groupBy(lambda p:p[0][0]).map(lambda p:p[0]).toLocalIterator()]
		final_list = zip(particles_ini,xbest_ini)
		xbest_red_1 =  min(final_list,key=lambda t:t[1])
		xworst_red_1 =  max(final_list,key=lambda t:t[1])
		for key,value in dict1.items():
			if (xbest_red_1[0] == key):
				best_centroids = value
		
		new_closest = data5.map(lambda p: (thedis(p, best_centroids)))
		locl_grp_op = new_closest.groupBy(lambda p:p[0]).map(lambda p:p[1])
		loc_comb = locl_grp_op.map(lambda x: [a[1] for a in x])
		locl_dist_fin = [x for x in loc_comb.map(lambda x: reduce(lambda a,b: numpy.sum([a, b], axis=0),x)).toLocalIterator()]
		clus_index_loc = [x for x in new_closest.groupBy(lambda p:p[0]).map(lambda p:p[0]).toLocalIterator()]
		loc_fin_dis = zip(clus_index_loc,locl_dist_fin)
		x_best_k = min(final_list,key=lambda t:t[1])
		chrg_grop = grp_particles.map(lambda x: [a for a in x])
		chrg_grop_2 = grp_particles.flatMap(lambda x: [a for a in x])
		denom = xworst_red_1[1] - xbest_red_1[1]
		q_cal = [x for x in chrg_grop.flatMap(lambda p:chargefun(p,x_best_k,denom)).toLocalIterator()]
		new_chrg_grop_2 = chrg_grop_2.map(lambda x:(x[0],x[1][0],x[1][1]))
		grop_2_cart = new_chrg_grop_2.cartesian(new_chrg_grop_2).map(lambda x:(x[0][0],(x[0][1],x[0][2],x[1][0],x[1][1],x[1][2],numpy.sqrt(numpy.sum(numpy.subtract(x[0][1], x[1][1]) ** 2))))).filter(lambda x: x[1][5] != 0.0)
		final_grop_2 = grop_2_cart.reduceByKey(lambda x, y: add_tuples_tolist(x, y)).map(lambda x: (x[0], min(x[1], key=lambda y:y[5])))
		map_tot_forc_col = final_grop_2.map(lambda p:dummyfun(p,q_cal))
		force_magnitude = map_tot_forc_col.map(lambda p:(p[0][0],((p[0][1][0]/numpy.sqrt(numpy.sum(p[0][1][0] ** 2, p[0][1][1] ** 2))),(p[0][1][1]/numpy.sqrt(numpy.sum(p[0][1][0] ** 2, p[0][1][1] ** 2))))))
		map_tot_forc_col.collect()
		print "**********"
		partcl_centroid.collect()
		print "*********"
		points_grp.collect()
		print "*****"
		combined_rdd = partcl_centroid.union(points_grp).reduceByKey(lambda x,y : x+y)
		combined_rdd.collect()
		print "##############################################################################################################################################################################################"
		partcile_limits = combined_rdd.map(lambda x:limitsfun(x))
		force_limits_rdd = force_magnitude.union(partcile_limits).reduceByKey(lambda x,y : x+y)
		End_iter_rdd = force_limits_rdd.map(lambda p:movement_fun(p)).collect()
		end_iter_key_list = []
		counter_key_end_iter = 0
		for key, group in groupby(sorted(End_iter_rdd, key=lambda k: k[0][0]),lambda x: x[0][0]):
			counter_key_end_iter = counter_key_end_iter + 1
		
		fin_arr_for_end_iter = [[]] * counter_key_end_iter 
		for key, group in groupby(sorted(End_iter_rdd, key=lambda k: k[0][0]),lambda x: x[0][0]):
			end_iter_key_list.append(key)
			conv_to_lis = list(group)
			new_temp_end_arr = []
			for i in range(len(conv_to_lis)):
				new_temp_end_arr.append([conv_to_lis[i][0][1],conv_to_lis[i][1]])
			fin_arr_for_end_iter[key] = new_temp_end_arr	
		
		dict1 = dict(zip(end_iter_key_list,fin_arr_for_end_iter))