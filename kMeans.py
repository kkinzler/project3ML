import numpy as np
import math
import matplotlib.pyplot as pl
import copy


#grab the data
data = np.loadtxt('GMM_data_fall2019.txt')

#verify data in np array
#print(data[:5])
#print(data[:5,1:])
#print(data[:5,:1])


#HYPER PARAMETERS
kClusters   = 6
randomStarts = 5
maxIterations = 500
wcStart = 1500.0

dataLength = data.shape[0]
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']	


def euclidDistance(x1, y1, x2, y2):
	return math.sqrt((x2-x1)**2 + (y2-y1)**2)


#run the kMeans algorithm with different values for k (eg. k = 2-10)
while kClusters > 1:
	# grab random data points for centroid of  each cluster
	initialCentroids = np.random.randint(low=0, high=dataLength,
										 size=(randomStarts, kClusters))
#	print("kClusters: ", kClusters)	
	newCentroids = [[] for i in range(kClusters)]
	randomStarts = 5 
	solutions = np.zeros(randomStarts + 1,dtype=float)
	bestWcss = 1500.0
	bestSolution = []

	# run 10 times with random initial centroids.
	while randomStarts > 0:
		#print("randomStarts: ", randomStarts)
		#print("iteration: ", iterations)
		ransomStarts = 5
		centroidIndices = initialCentroids[randomStarts - 1]
		print("cIndices: ", centroidIndices)
		iterations = 0
		found = False

		#store new value for centroid indices
		centroids = []
		for index in centroidIndices:
			centroids.append(data[index])

		while found == False: 
			
			# List of minimum centroids for each datum (datum index -> centroid index)
			distances = []
			for datum in data:
				centdists = []
				for centroid in centroids:
					centdists.append(euclidDistance(datum[0], datum[1], centroid[0], centroid[1]))
				minindex = centdists.index(min(centdists))
				distances.append({"datum": datum, "centroidindex": minindex, "distance": centdists[minindex]})


			for i in range(kClusters):
				#store each data point in respective cluster
				inCluster = [[] for i in range(kClusters)]
				for row in distances:
					if row["centroidindex"] == i:
						inCluster[i].append(row["datum"])
				#calculate new mean
				newCentroids[i] = np.mean(inCluster[i], axis=0)
				

			if maxIterations == iterations:
				for k in range(kClusters):
					for row in distances:
						if row["centroidindex"] == k:
							solutions[k] += row["distance"]
				wcss = np.sum(solutions)		
				if wcss < bestWcss:
					bestSolution = copy.deepcopy(distances)
					
				#print("randomStarts: ", randomStarts)
				#print("randomSolutions:  ", tenRandomSolutions)	
				randomStarts -= 1
				found = True


			else:
				centroids = copy.deepcopy(newCentroids)
				"""
				if iterations % 100 == 0:
					for i in range(kClusters):
						for row in distances:
							if distances["minindex"] == i:
								pl.scatter(distances["datum"][0],distances["datum"][1],c=color[i],marker='+')

					pl.title('kmean cluster')
					pl.show()

				"""	
				iterations += 1

	for i in range(kClusters):
		for row in distances:
			if row["centroidindex"] == i:
				pl.scatter(row["datum"][0],row["datum"][1],c=color[i],marker='+')

	pl.title('kmean cluster')
	pl.show()
	
	kClusters -= 1
