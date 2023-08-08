# Load the iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# Select the variables you want to cluster on
iris_cluster = iris.data[:, [0, 1, 2, 3]]

# Perform K-means clustering with 3 clusters
from sklearn.cluster import KMeans
kmeans_results = KMeans(n_clusters=3, random_state=123).fit(iris_cluster)

# Print the cluster assignments
print(kmeans_results.labels_)
