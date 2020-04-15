import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

''' sklearn.cluster.AgglomerativeClustering
linkage{“ward”, “complete”, “average”, “single”}, default=”ward”
Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. 
The algorithm will merge the pairs of cluster that minimize this criterion.
. ward minimizes the variance of the clusters being merged.
. average uses the average of the distances of each observation of the two sets.
. complete or maximum linkage uses the maximum distances between all observations of the two sets.
. single uses the minimum of the distances between all observations of the two sets.


affinity: str or callable, default=’euclidean’
Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”. 
If linkage is “ward”, only “euclidean” is accepted. 
If “precomputed”, a distance matrix (instead of a similarity matrix) is needed as input for the fit method.
'''
## results of AgglomerativeClustering
'''children_:The children of each non-leaf node.  Values less than n_samples correspond to leaves of the tree which are the original samples. 
A node i greater than or equal to n_samples is a non-leaf node and has children is children_[i - n_samples]. 
Alternatively at the i-th iteration, children[i][0] and children[i][1] are merged to form node n_samples + i'''

def cal_linkage_matrix(model):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_) #150 samples
    for i, merge in enumerate(model.children_):
        # print(i,merge)
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples: # leaf node
                current_count += 1
            else: # cluster head or merged node
                current_count += counts[child_idx - n_samples] #count all nodes belongs to this head ??? strange form?
        counts[i] = current_count
        # print(counts)

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    return n_samples, linkage_matrix


def retrieve_leaves(cluster_head, n_samples=150):
    leaves = []
    if(cluster_head < n_samples):
        leaves.append(cluster_head)
    else:
        for idx in model.children_[cluster_head - n_samples]:
            leaves = leaves + retrieve_leaves(idx)
    return leaves

def retrieve_cluster_head(cluster_head, n_samples=150):
    if (cluster_head >= n_samples):
        return model.children_[cluster_head - n_samples]


iris = load_iris()
X = iris.data

# setting distance_threshold=0 ensures we compute the full tree.
result = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

### OR Plot Only get labels
# model1 = result.fit_predict(X)
# print(model1)

#OR Plot
model = result.fit(X)
print(model.labels_)
# print(model.children_)
# print(model.distances_)

numb_samples, rs_linkage_matrix = cal_linkage_matrix(model)

# Plot the corresponding dendrogram
# plt.title('Hierarchical Clustering Dendrogram')
# change p value to 5 if we want to get 5 levels
# rs_dendrogram = dendrogram(rs_linkage_matrix, truncate_mode='level', p=3)
#
# print(rs_dendrogram['ivl']) #x_axis of dendrogram => index of nodes or (Number of points in clusters (i))
# print(rs_dendrogram['leaves']) # merge points
# plt.xlabel("index of nodes or (Number of points in clusters (i)).")

K_Levels = 4
for level in range(1,K_Levels):
    print("=> GENERALIZED LEVEL", K_Levels - level, "-------")
    rs_dendrogram = dendrogram(rs_linkage_matrix, truncate_mode='level', p=level)
    cluster_heads = rs_dendrogram['leaves']
    for g_idx in cluster_heads:
        if (g_idx < numb_samples):
            print("Leaf:", g_idx)
        else:
            if(level == K_Levels -1):
                print("All leaves:",retrieve_leaves(g_idx, numb_samples))
            else:
                print("Children:",retrieve_cluster_head(g_idx, numb_samples))






