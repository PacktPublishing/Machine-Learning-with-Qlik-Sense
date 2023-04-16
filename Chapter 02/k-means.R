# Load the iris dataset
data(iris)

# Select the variables you want to cluster on
iris_cluster <- iris[, c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")]

# Perform K-means clustering with 3 clusters
set.seed(123)
kmeans_results <- kmeans(iris_cluster, centers = 3)

# Print the cluster assignments
kmeans_results$cluster
