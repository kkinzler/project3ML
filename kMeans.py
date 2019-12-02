import numpy as np
import math


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

#run the kMeans algorithm with different values for k (eg. k = 2-10)
while kClusters > 1:
	#grab random data points for centroid of  each cluster
	randomStarts = 10
	#maxIter = 5000
	poison = True	
	initCenter = np.random.randint(low = 0, high = dataLength, size =(randomStarts, kClusters))
	print("kClusters: ", kClusters)	

	#run 10 times with random initial centroids. 
	while randomStarts > 0:
		centroids = initCenter[randomStarts - 1]

		#run algorithm until the centroids don't change or
		#the max number of iterations has been spent
		#while isClustering(centroids, prevCentroids, iteration):
		while poison:
			poison = False
			#assignment = np.zeros(data.shape[0])
			#for row in data:
				#assignment[row] = **min(euclideanDistance(centroid(x), row))
			#for row in assignment:
				#find mean of each cluster
					
		
		#store result from each randomStart in array	
		randomStarts = randomStarts - 1

	#store best solution from each randomStart in array for each number of clusters
	kClusters = kClusters - 1

#print the plots

#TODO:
#	1)calculate the sum of squared difference between each data point and centroid
#	2)assign each data point to the centroid it is closest to
#	3)recompute centroids for each cluster by taking average of all points in cluster	
#	4)write method to check whether centroids have changed from last iteration
#	5)store each 'randomStarts' solution and pick the smallest for each number of 'k'
#	6)plot the initial points and the resulting cluster (easy peasy, look online)