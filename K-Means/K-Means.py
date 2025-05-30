#-*-codid:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset =pd.read_csv('D:\software\git\machine_learning\K-Means\Customer_Info.csv')
x=dataset.iloc[:,[4,3]].values

sumDS =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)
    sumDS.append(kmeans.inertia_)
    #print(kmeans.inertia_)
plt.plot(range(1,11), sumDS)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters K')
plt.ylabel('SSE')
plt.show()

kmeans = KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_kmeans = kmeans.fit_predict(x)
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1',marker='^')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2', marker='o')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3', marker='s')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=250, c='yellow', label='Centroids')
plt.title('Clusters of customers Into')
plt.xlabel('Deposit')
plt.ylabel('Age')
plt.legend()
plt.show()