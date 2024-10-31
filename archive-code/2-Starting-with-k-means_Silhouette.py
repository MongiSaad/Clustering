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
from sklearn.metrics import silhouette_score

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="aggregation.arff"
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

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")

stat = []
for k in range(2, 10):
    model = cluster.KMeans(n_clusters=k, init='k-means++', n_init=1)
    model.fit(datanp)
    labels = model.labels_
    score = silhouette_score(datanp, labels)
    stat.append([k, score])

#plt.figure(figsize=(6, 6))
#plt.scatter(f0, f1, c=labels, s=8)
#plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
#plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
#plt.show()


k_tab =  [row[0] for row in stat]
db_tab = [row[1] for row in stat]
k_optimal = np.argmax(db_tab) + 2
plt.plot(k_tab, db_tab)
plt.xlabel("nb cluster")
plt.ylabel("Silhouette Value")
plt.title("Silhouette Method: k_optimal = "+ str(k_optimal))
plt.axvline(x=k_optimal, color="orange")
plt.show()

#print("labels", labels)



# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(data, labels)
print(f"Davies-Bouldin Index: {db_index}")



