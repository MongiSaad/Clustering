import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


###################################################################
# Exemple : Agglomerative Clustering


path = './artificial/'
name="aggregation.arff"

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

linkages = ["single", "complete", "ward", "average"]
stats = [[], [], [], []]

for linkage in range(4): 
    for k in range(2,20):
        tps1 = time.time()
        model = cluster.AgglomerativeClustering(linkage=linkages[linkage], n_clusters=k)
        model = model.fit(datanp)
        tps2 = time.time()
        labels = model.labels_
        score = metrics.davies_bouldin_score(datanp, labels)
        stats[linkage].append([k, score])

    k_tab =  [row[0] for row in stats[linkage]]
    db_tab = [row[1] for row in stats[linkage]]
    plt.plot(k_tab, db_tab, label = linkages[linkage])

#######################################################################

# Déterminer quel linkage a le meilleur résultat
stats = np.array(stats)
stats_min = [np.min(row[:, 1]) for row in stats]
print("stats_min : ", stats_min)
best = np.argmin(stats_min)
row_best = [row[1] for row in stats[best]]
k_optimal = np.argmin(row_best) + 2


plt.xlabel("nb cluster")
plt.ylabel("Davies Bouldin Value")
plt.title("Davies Bouldin Method: k_optimal : " + str(k_optimal) + ", linkage optimal : " + linkages[best])
plt.axvline(x=k_optimal, color="purple", label = "K optimal")
plt.legend()
plt.show()