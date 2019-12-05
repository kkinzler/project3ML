import numpy as np
import math
import matplotlib.pyplot as pl


#grab the data
data = np.loadtxt('GMM_data_fall2019.txt')


#HYPER PARAMETERS(ish)
kClusters     = 6
maxIterations = 500
randomStarts  = 6
m             = 2.0

dataLength = data.shape[0]
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']	

pl.scatter(data[0:,0],data[0:,1],c=color[0],marker='+')
pl.title('fuzzy c-mean cluster')
pl.show()


def findCentroid(data, weight, m):
	#num = numerator
	#den = denominator
	xNum = 0.0
	yNum = 0.0
	den = 0.0

	for row in range(data.shape[0]):
		weightSquared = weight[row]**2
		xNum += weightSquared * data[row,0]
		yNum += weightSquared * data[row,1]
		den += weightSquared				
	return [xNum/den, yNum/den]

def calcDistance(data, centroids):
	#print("data.shape[0]   ",data.shape[0])
	#print("\ndata[:5]   ", data[:5])
	#print("\ncentroids   ",centroids)
	distance = np.zeros((data.shape[0],kClusters),dtype=float)	
	for row in range(data.shape[0]):
		for centroid in range(centroids.shape[0]):
			x = centroids[centroid,0] - data[row,0]
			y = centroids[centroid,1] - data[row,1]
			x = np.power(x,2)
			y = np.power(y,2)

			value = x + y
			distance[row,centroid] = np.sqrt(value)	
	
	return distance

#because m == 2 theres no need to calculate the power with each iteration of 
#the summation because the power is just 1/(m-1) => 1/(2-1) => 1
#that functionality would need to be added if you want to change the value of
#m from 2
def calcMembership(distance):
	fuzzy = np.zeros((data.shape[0],kClusters),dtype=float)
	for row in range(distance.shape[0]):
		den = 0.0
		for col in range(kClusters):
			den += 1 / distance[row,col]
		for col in range(kClusters):
			fuzzy[row,col] = (1/distance[row,col]) / den
	return fuzzy

def euclidDistance(x1, y1, x2, y2):
	return math.sqrt((x2-x1)**2 + (y2-y1)**2)


def calcWCSS(data, membership, centroids):
	iter = 0
	eachWCSS = np.zeros(kClusters,dtype=float)

	for row in membership:
		max = np.argmax(row)
		eachWCSS[max] += euclidDistance(data[iter,0], data[iter,1], centroids[max,0], centroids[max,1])
	
	return np.sum(eachWCSS)	


#need to think through how to hold the lowest sum of squares solution 
bestWCSS = 1500.0


for starts in range(2, randomStarts + 1):
	bestWCSS = math.inf
	bestMembership = []
	kClusters = starts
	#store each clusters centroid on each iteration	
	clusterCentroids = np.zeros((kClusters, 2),dtype=float)	

	#store distance from data point to each clusters centroid	
	distanceToCentroids = np.zeros((data.shape[0],kClusters),dtype=float)	

	#add membership weights to each data point for every cluster
	#sum of all weights for a data point must sum to 1
	membership = np.random.random_sample((data.shape[0], kClusters))
	for row in range(data.shape[0]):
		membership[row] = np.random.dirichlet(np.ones(kClusters),size=1)
	#print("initial membership:\n",membership[:5])

	#add membership weights to data
	#data = np.append(data, membership, 1)

	for iteration in range(maxIterations):
		
		#calculate the cluster centroids
		for cluster in range(kClusters):
			clusterCentroids[cluster] = findCentroid(data, membership[0:,cluster], m)	
			
	
		#calculate distances from each point to each cluster
		distanceToCentroids = calcDistance(data, clusterCentroids)
		
		#calculate which cluster each data point is a member of
		membership = calcMembership(distanceToCentroids)	

		'''
		if iteration % 100 == 0:
			iter = 0
			for row in membership:
				pl.scatter(data[iter,0],data[iter,1],c=color[np.argmax(row)],marker='+')
				iter += 1
			pl.title('fuzzy c-mean cluster')
			pl.show()
		'''
	#time to calculate the sum of square for the solution and save the best one
	wcss = calcWCSS(data, membership, clusterCentroids)

	if wcss < bestWCSS:
		bestWCSS = wcss
		bestMembership = membership.copy()

	print("Model for " + str(starts) + " clusters")
	print("Model error: " + str(wcss))
	iter = 0
	for row in bestMembership:
		pl.scatter(data[iter,0],data[iter,1],c=color[np.argmax(row)],marker='+')
		iter += 1
	pl.title('fuzzy c-mean cluster')
	pl.show()





