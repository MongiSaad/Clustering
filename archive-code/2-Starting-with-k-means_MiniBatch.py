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
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="birch-rg3.arff"

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

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
timeMiniBatch = []
for k in range(2,50):
    tps1 = time.time()
    model = cluster.MiniBatchKMeans(init ='k-means++', n_clusters = k, n_init = 1)
    model.fit(datanp)
    tps2 = time.time()
    labels = model.labels_
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_
    runtime = round((tps2 - tps1)*1000,2)
    timeMiniBatch.append(runtime)

#print("Runtime using MinibatchKMeans: ", str(runtime),"ms")

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()



# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
timeKMeans = []
for k in range(2,50):
    tps3 = time.time()
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    tps4 = time.time()
    labels = model.labels_
    # informations sur le clustering obtenu
    iteration = model.n_iter_
    inertie = model.inertia_
    centroids = model.cluster_centers_
    runtime = round((tps4 - tps3)*1000,2)
    timeKMeans.append(runtime)

#print("Runtime using KMeans: ", str(runtime),"ms")
#print("labels", labels)
plt.figure(figsize=(10, 6))  # Optionnel : Ajustez la taille de la figure
plt.plot(range(2, 50), timeKMeans, color='green', label='Temps d\'exécution K-means')
plt.plot(range(2, 50), timeMiniBatch, color='blue', label='Temps d\'exécution minibatch K-means')  # Assurez-vous que timeMiniBatch est défini
plt.title('Comparaison des temps d\'exécution de K-means et minibatch K-means')
plt.xlabel('Nombre de clusters (k)')
plt.ylabel('Temps d\'exécution (ms)')
plt.legend()
plt.grid()  # Optionnel : Ajoute une grille pour une meilleure lisibilité
plt.xlim(2, 49)  # Définir les limites pour l'axe des x si nécessaire
plt.ylim(0, max(max(timeKMeans), max(timeMiniBatch)) * 1.1)  # Définir les limites pour l'axe des y pour une meilleure visualisation
plt.show()

from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(centroids)
print(dists)

