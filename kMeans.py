import numpy as np
import math
#import matplotlib.pyplot as plt


#grab the data
data = np.loadtxt('GMM_data_fall2019.txt')

#verify data in np array
#print(data[:5])
#print(data[:5,1:])
#print(data[:5,:1])


#HYPER PARAMETERS
kClusters   = 10
randomStarts = 10
maxIterations = 5000

dataLength = data.shape[0]

poison = True


def euclidDistance(x1, y1, x2, y2):
	return math.sqrt((x2-x1)**2 + (y2-y1)**2)


# run the kMeans algorithm with different values for k (eg. k = 2-10)
while kClusters > 1:
	# grab random data points for centroid of  each cluster
	randomStarts = 10
	# maxIter = 1000
	poison = True	
	initialCentroids = np.random.randint(low=0, high=dataLength,
										 size=(randomStarts, kClusters))
	print("kClusters: ", kClusters)	
	
	iterations = 0
	newCentroidIndices = [[] for i in range(kClusters)]
	tenRandomSolutions = [[] for i in range(0, 10)]
	# run 10 times with random initial centroids.
	while randomStarts > 0:
		centroidIndices = initialCentroids[randomStarts - 1]
		#store new value for centroid indices
		centroids = []
		for index in centroidIndices:
			centroids.append(data[index])
		# List of minimum centroids for each datum (datum index -> centroid index)
		distances = []
		for datum in data:
			centdists = []
			for centroid in centroids:
				centdists.append(euclidDistance(datum[0], datum[1], centroid[0], centroid[1]))
			minindex = centdists.index(min(centdists))
			#---what's the difference between minindex and centdists[minindex]---
			distances.append({"datum": datum, "centroidindex": minindex, "distance": centdists[minindex]})


		for i in range(kClusters):
			#store each data point in respective cluster
			inCluster = [[] for i in range(kClusters)]
			for row in distances:
				if row["centroidindex"] == i:
					inCluster[i].append(row["datum"])
			#calculate new mean
			newCentroidIndices[i] = np.mean(inCluster[i], axis=0)


		if (centroidIndices == newCentroidIndices) || maxIterations == iterations:
			stop = True	
			tenRandomSolutions.append(newCentroidIndices) 
		else:
			centroidIndices = newCentroidIndices
	
	for row in tenRandomSolutions:
		for datum in data:

	for i in range(dataLength):
		if iris.target[i] == 0:
		c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
		elif iris.target[i] == 1:
		c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
		elif iris.target[i] == 2:
		c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')
	pl.legend([c1, c2, c3], ['Setosa', 'Versicolor','Virginica'])
	pl.title('Iris dataset with 3 clusters and known outcomes')
	pl.show()


