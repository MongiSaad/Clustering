import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

##################################################################
# Exemple : DBSCAN Clustering


path = './artificial/'
name="donut3.arff"

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

####################################################
# Standardisation des donnees

scaler = preprocessing.StandardScaler().fit(datanp)
data_scaled = scaler.transform(datanp)
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne

#plt.figure(figsize=(10, 10))
plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")
plt.show()


#####################################################
# Recherche de eps

# Distances aux k plus proches voisins
k=4
neigh = NearestNeighbors(n_neighbors=k) 
neigh.fit(data_scaled)
distances , indices = neigh.kneighbors(data_scaled)
# distance moyenne sur les k plus proches voisins
# en retirant le point "origine"
newDistances = np.asarray([np.average(distances[i][1:]) for i in range(0,distances.shape[0])])
# trier par ordre croissant
distancetrie = np.sort(newDistances)
plt.title("Plus proches voisins "+str(k)) 
plt.plot(distancetrie)
plt.show()

# On récupère eps
print("entrer epsilon : ")
eps = float(input())

stat = []
for min_pts in range(3, 10):

    tps1 = time.time()
    model = cluster.DBSCAN(eps=eps, min_samples=min_pts)
    model.fit(data_scaled)

    tps2 = time.time()
    labels = model.labels_

    score = metrics.calinski_harabasz_score(data_scaled, labels)
    stat.append([min_pts, score])


min_pts_tab =  [row[0] for row in stat]
score_tab = [row[1] for row in stat]
min_samples_optimal = np.argmax(score_tab) + 3
plt.plot(min_pts_tab, score_tab)
plt.xlabel("min_samples")
plt.ylabel("Calinski Harabasz Value")
plt.title("Calinski Harabasz Method: min_samples_optimal = "+ str(min_samples_optimal))
plt.axvline(x=min_samples_optimal, color="purple")
plt.show()