"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="cure-t0-2000n-2D.arff"
test = ""

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne
'''
#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()'''

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")

stat = []
for k in range(1, 50):
    tps1 = time.time()
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_
    stat.append([k, iteration, inertie, round((tps2 - tps1)*1000,2)])

#plt.figure(figsize=(6, 6))
#plt.scatter(f0, f1, c=labels, s=8)
#plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
#plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
#plt.show()

cond = 1
indice = 0
div_prev = 0
k_temp = 0
while(cond > 0):
    diff1 = stat[indice][2] - stat[indice + 1][2]
    diff2 = stat[indice + 1][2] - stat[indice + 2][2]
    div = diff1 / diff2
    k_temp = stat[indice][0]
    cond = div - div_prev
    div_prev = div
    indice += 1

k_optimal = k_temp

print("résultat optimal : ", k_optimal)

k_tab =  [row[0] for row in stat]
inertie_tab = [row[2] for row in stat]
plt.plot(k_tab, inertie_tab)
plt.xlabel("nb cluster")
plt.ylabel("Inertie")
plt.title("Nombre de cluster optimal : "+ str(k_optimal))
plt.axvline(x=k_optimal, color="red")
plt.show()

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)

from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(centroids)
print(dists)
