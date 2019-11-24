import numpy as np
import random
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from functools import reduce
from itertools import repeat
from pyspark import SparkContext, SparkConf

def Partition_Data(raw_data, num_partitions):
    '''args:
        raw_data(numpy array): (n,d) array where n is # of points and d is # of dimensions
        num_partitions(integer): number of data partitions
       returns:
        partitioned_data(list): list of data partitions(d dimensional numpy arrays)'''
    index = np.random.randint(low = 1, high = num_partitions+1, size = len(raw_data))
    partitioned_data = []
    
    for k in range(1,num_partitions+1):
        partitioned_data.append(raw_data[index == k])
    
    return partitioned_data


def Local_DBSCAN(data):
    '''args:
        data(numpy array): (n,d) size data for clustering
        min_pts, eps: parameters for dbscan algorithm
       returns:
        labels(numpy array): (n,) array indexing cluster label of each point in dataset'''
    min_pts = MIN_PTS/num_partitions
    eps = EPS
    db = DBSCAN(eps=eps, min_samples=min_pts).fit(data)
    return db.labels_

def Plot_Clusters(data, labels):
    '''args:
        data(numpy array): (n,d) sized data
        labels(numpy array): (n,) corresponding cluster for each point'''
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Plot result
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    

def Calculate_Centroids(data_and_labels):
    data = data_and_labels[0]
    labels = data_and_labels[1]
    '''args:
        data(numpy array): (n,d) sized data
        labels(numpy array): (n,) corresponding cluster for each point
       returns:
        centroids(list): list of centroids (d,) corresponding to each cluster in data'''

    unique_labels = set(labels)
    centroids = []
    for k in unique_labels:        
        if k != -1:
            xy = data[labels == k] #class mask where labels == k
            centroid = np.mean(xy, axis=0)
            centroids.append(centroid)
        
    return centroids

def Find_min_d(centroids):
    '''args:
        centroids(list): list of centroids (d,) corresponding to each cluster in data
       returns:
        min_d(float): minimum distance between any two clusters in the data'''
    dist = []
    min_d = np.inf
    for i in range(len(centroids) - 1):
        for j in range(1, len(centroids)):
            if i != j:
                d = np.linalg.norm(centroids[i] - centroids[j])
                if d < min_d:
                    min_d = d
    
    return min_d
            

def Merge_Partitions(data1, data2):
    '''args:
        data1(tuple): dataset from a partition containing dataset (n, d), centroids(list) and
                      labels(n,) in that order
        data2(tuple): same as data1
       returns:
         merged tuple of data1 and data2'''

    partition1 = data1[0]
    centroids1 = data1[1]
    labels1 = data1[2]
    partition2 = data2[0]
    centroids2 = data2[1]
    labels2 = data2[2]
    
    for i in range(len(centroids1)):
        for j in range(len(centroids2)):
            if j in labels2 and np.linalg.norm(centroids1[i] - centroids2[j]) < SIGMA:
                #combine labels1[i] and labels2[j]
                partition1 = np.concatenate((partition1, partition2[labels2 == j]))
                partition2 = partition2[labels2 != j]
                labels1 = np.concatenate((labels1, np.repeat(i, len(labels2[labels2 == j]))))
                labels2 = labels2[labels2 != j]
                centroids1[i] = (centroids1[i]*len(labels1[labels1 == i])
                                + centroids2[j]*len(labels2[labels2 == j]))/(len(labels1[labels1 == i])
                                +len(labels2[labels2 == j]))
    #Add all the noise points
    partition1 = np.concatenate((partition1, partition2[labels2 == -1]))
    partition2 = partition2[labels2 != -1]
    labels1 = np.concatenate((labels1, labels2[labels2 == -1]))
    labels2 = labels2[labels2 != -1]
        
    if len(labels2) == 0:
        return (partition1, centroids1, labels1)
    
    max_i = np.max(labels1)
    j = 1
    
    for i in set(labels2):
        partition1 = np.concatenate((partition1, partition2[labels2 == i]))        
        labels1 = np.concatenate((labels1, np.repeat(max_i + j,len(labels2[labels2 == i]))))
        centroids1.append(centroids2[i])
        j = j+1
        
    return (partition1, centroids1, labels1)
        


if __name__ == "__main__":
    
    num_partitions = 4
    conf = SparkConf().setAppName("S_DBSCAN").setMaster("local[4]")
    sc = SparkContext(conf = conf)
    
    #Generate sample data
    centers = [[10, 10], [-10, -10], [10, -10]]
    X, labels_true = make_blobs(n_samples=7500, centers=centers, cluster_std=0.4,
                                random_state=0)    
    dummy = StandardScaler().fit_transform(X)    
    
    #Local DBScan
#    db = Local_DBSCAN(dummy, min_pts = 10, eps=0.3)
#    Plot_Clusters(dummy, db.labels_)
#    
    
    partitioned_data = Partition_Data(dummy, num_partitions)
    
    partition_rdd = sc.parallelize(partitioned_data)
    #plot partitioned data
#    plt.scatter(partitioned_data[1][:,0], partitioned_data[1][:,1])
#    plt.show()
    
    MIN_PTS, EPS = 10, 0.3
#    db = list(map(Local_DBSCAN, partitioned_data, repeat(min_pts/num_partitions), repeat(eps))) 
    db_labels = partition_rdd.map(Local_DBSCAN).collect()
    data_and_labels = []
    for i in range(num_partitions):
        data_and_labels.append((partitioned_data[i], db_labels[i]))
    
    data_and_labels_rdd = sc.parallelize(data_and_labels)
    centroids_rdd = data_and_labels_rdd.map(Calculate_Centroids)
    centroids = centroids_rdd.collect()
    #Find the minimum distance b/w clusters within each partition
    min_d = centroids_rdd.map(Find_min_d).collect()
    
    #Find the global minimum distance
    min_D = np.min(min_d)
    #set the maximum distance for clusters to merge
    SIGMA = min_D/2
    
    all_data = []
    for i in range(num_partitions):
        all_data.append((partitioned_data[i], centroids[i], db_labels[i]))
    
    all_data_rdd = sc.parallelize(all_data)
    result = all_data_rdd.reduce(Merge_Partitions)
    Plot_Clusters(result[0], result[2])
    sc.stop()
    
    
    