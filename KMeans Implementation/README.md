

# k Means - Implementation in Python
---

The `kmeans` class is an implementation of the k-means clustering algorithm in Python. It can be used to cluster data points into k clusters based on their similarity.

## Training
---

The `kmeans` class uses the following steps to train the model:

1. Initialize k cluster centroids randomly.
2. Assign each data point to the nearest centroid.
3. Compute the sum of squared errors (SSE) for each cluster.
4. Update the centroids by computing the mean of all data points assigned to each cluster.
5. Repeat steps 2-4 until the SSE converges or a maximum number of iterations is reached.

## Prediction
---

The `kmeans` class provides a `predict` function that can be used to predict the cluster labels for new data points. The function works as follows:

1. Assign each new data point to the nearest cluster centroid based on the Euclidean distance.
2. Return the cluster label of the assigned centroid.

## Other Functions
---

The `kmeans` class also provides the following functions:

- `initialiseClusterCentroids`: Initializes the cluster centroids randomly.
- `ComputeSSE`: Computes the sum of squared errors (SSE) for each cluster.
- `computeNewCentroids`: Computes the new centroids by computing the mean of all data points assigned to each cluster.
- `assignNewClusters`: Assigns each data point to the nearest cluster centroid based on the Euclidean distance.
- `trainkMeans`: Trains the k-means model by updating the centroids until convergence.
- `fit`: Fits the k-means model to the input data and returns the cluster labels for each data point.
- `predictMisclassification`: Computes the number of misclassifications for each cluster.
- `historyPlotter`: Plots the SSE history over the training iterations.
- `getCentroid`: Returns the centroid coordinates for each cluster.

These functions can be used to analyze the trained model and visualize the clustering results.

### Accuracy Calculation 
---

GitHub Copilot: Here's a section on the logic code of the `accuracyComputation` function for the README in list format:

- The `accuracyComputation` function computes the accuracy score for the predicted labels based on all possible permutations of the true labels.
- It takes two input arrays, `original` and `predicted`, which represent the true labels and the predicted labels, respectively.
- The function first computes the number of unique classes in the `original` array.
- It generates all possible permutations of the class labels using the `permutations` function from the `itertools` module.
- It then iterates through each permutation and replaces the predicted labels with the corresponding permutation.
- For each permutation, the function computes the accuracy score using the `accuracy_score` function from the `sklearn.metrics` module.
- It stores the accuracy score for each permutation in the `ACCURACYSCORE` list.
- It also stores the predicted labels for each permutation in the `ACCURACYARRAY` list.
- Finally, the function returns the maximum accuracy score and the corresponding predicted labels from the `ACCURACYARRAY` list.