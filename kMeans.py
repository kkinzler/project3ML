import numpy as np
import math
import matplotlib.pyplot as pl
import copy

# grab the data
data = np.loadtxt('GMM_data_fall2019.txt')

# verify data in np array
# print(data[:5])
# print(data[:5,1:])
# print(data[:5,:1])


# HYPER PARAMETERS
# Initial number of clusters (reduces by 1 each run)
KCLUSTERS = 6
# Number of times each of the clusters are reset to
# random values each run
RANDOMSTARTS = 5
# Number of times the k-means are recalculated per random run
MAXITERATIONS = 500

wcStart = 1500.0

dataLength = data.shape[0]
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def euclidDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# run the kMeans algorithm with different values for k (eg. k = 2-10)
kClusters = KCLUSTERS
while kClusters > 1:
    print("Finding means for : " + str(kClusters) + " clusters")
    # grab random data points for centroid of  each cluster
    randomStarts = RANDOMSTARTS
    initialCentroids = np.random.randint(low=0, high=dataLength, size=(randomStarts, kClusters))
    # print("kClusters: ", kClusters)

    newCentroids = [[] for i in range(kClusters)]

    solutions = np.zeros(randomStarts + 1, dtype=float)
    bestWcss = wcStart
    bestSolution = []

    # run 10 times with random initial centroids.

    while randomStarts > 0:
        # Zero out solutions array
        solutions = np.multiply(0, solutions)
        print("Starting random iteration " + str(randomStarts))
        # Get a set of random centroids
        centroidIndices = initialCentroids[randomStarts - 1]
        iterations = 0
        # Is best solution found yet?
        found = False

        # store new values for centroid indices
        centroids = []
        for index in centroidIndices:
            centroids.append(data[index])
        # While best solution has not yet been found...
        while not found:
            distances = []
            # For each datum, compute it's euclidian distance from each of the
            # centroids, and store the smallest one in the distances list
            for datum in data:
                centdists = []
                for centroid in centroids:
                    centdists.append(euclidDistance(datum[0], datum[1], centroid[0], centroid[1]))
                minindex = centdists.index(min(centdists))
                distances.append({"datum": datum, "centroidindex": minindex, "distance": centdists[minindex]})

            # For each centroid, find the average distance from each datum which
            # listed it as having the smallest distance from it
            for i in range(kClusters):
                # store each data point in respective cluster
                inCluster = [[] for i in range(kClusters)]
                for row in distances:
                    if row["centroidindex"] == i:
                        inCluster[i].append(row["datum"])
                # calculate new mean
                newCentroids[i] = np.mean(inCluster[i], axis=0)

            if MAXITERATIONS == iterations:
                for k in range(kClusters):
                    for row in distances:
                        if row["centroidindex"] == k:
                            solutions[k] += row["distance"]
                # Sum of squares error for model
                wcss = np.sum(solutions)

                # print("Sum of squares error for model: ", wcss)
                if wcss < bestWcss:
                    bestSolution = copy.deepcopy(distances)

                # print("randomStarts: ", randomStarts)
                # print("randomSolutions:  ", tenRandomSolutions)
                randomStarts -= 1
                found = True

            else:
                centroids = copy.deepcopy(newCentroids)
                iterations += 1
    print("Best sum of squares error for model: ", wcss)
    for i in range(kClusters):
        for row in distances:
            if row["centroidindex"] == i:
                pl.scatter(row["datum"][0], row["datum"][1], c=color[i], marker='+')

    pl.title('kmean cluster')
    pl.show()

    kClusters -= 1
