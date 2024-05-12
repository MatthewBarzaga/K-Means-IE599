import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os

#function used to create an array of x and y coords given our text file input
def x_y(file_path):
    x_coords = []
    y_coords = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = line.split()
            x_coords.append(float(x))
            y_coords.append(float(y))
    return x_coords, y_coords

#helper function to calculate distance given two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

#helper function to calculate the Sum of Squared Error of any number of given clusters and centroids
def calculate_sse(centroids, clusters):
    sse = 0
    #this is saying for each cluster, we look at it's respective centroid
    for i, cluster in enumerate(clusters):
        centroid = centroids[i]
        #for each point in that cluster we sum up the distance of every point to the centroid and save it in our SSE.
        for point in cluster:
            sse += euclidean_distance(point, centroid) ** 2
    return sse

#function to implement k-means++
def k_means_plusplus(points, k):
    centroids = []
    centroids.append(random.choice(points))  # Initialize the first centroid randomly

    # Initialize the remaining k - 1 centroids
    for _ in range(k - 1):
        # here we calculate the  distance each point has to its nearest centroid
        distances = [min([euclidean_distance(point, centroid) for centroid in centroids]) for point in points]
        # then we select the max of these points, which effectively gives us the point which is furthest from any centroid
        # and we add that point as a new centroid
        max_distance_idx = np.argmax(distances)
        centroids.append(points[max_distance_idx])
    return centroids

#function to implement the majority of the k-means algorithm
def k_means(points, k, termination_criteria, sse_improvement, max_iterations, selection):
    if selection == 'R':
        #if random initilization we simply select random points to start with
        centroids = random.sample(points, k)
    elif selection == 'K':
        #if K++ initilization we call the k_means_plusplus function to setup our centroids
        centroids = k_means_plusplus(points, k)
    #we create an empty list of clusters, and set our SSE to 0
    clusters = [[] for _ in range(k)]
    sse = 0
    iteration = 0
    #this is the main loop of the algorithm, it will continue until our termination criteria is met
    while True:
        #this is where we assign each point to a cluster based on which centroid it is closest to
        for point in points:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid_idx = np.argmin(distances)
            clusters[closest_centroid_idx].append(point)
        #here we calculate the new centroids by taking the mean of all the points in each cluster
        centroids = [np.mean(cluster, axis=0) for cluster in clusters]
        new_sse = calculate_sse(centroids, clusters)
        #depending on which termination criteria the user chose, we check if we should break out of the loop
        if termination_criteria == '1':
            if iteration >= max_iterations:
                sse = calculate_sse(centroids, clusters)
                break
        
        elif termination_criteria == '2': 
            if iteration == 0:
                sse_improvement_percentage = 1
            #if we see our improvement become zero we know that we can break out as the algorithm is finshed
            elif sse_improvement == 0:
                if sse == new_sse:
                    break
            #otherwise we go until our improvement percent is lower than the bound we initially set
            elif sse != 0:
                sse_improvement_percentage = ((sse - new_sse) / sse )
                if sse_improvement_percentage < sse_improvement:
                        break
            sse = new_sse
        iteration += 1
        #we reset our clusters to empty lists so we can reassign points to them with their new centroids
        clusters = [[] for _ in range(k)]
    return centroids, clusters, sse, iteration

##function for creating a graph with  matplotlib.pyplot library
def visualize_clusters(centroids, clusters):
    # figure size
    plt.figure(figsize=(6.4, 4.8))

    # this selects a specific color scheme that I thought looked nice
    cmap = plt.get_cmap('tab20')  # Use a predefined color map
    num_clusters = len(clusters)
    colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]  # Generate distinct colors

    # Our cluster objects have an x and y array, it would look like this: cluster1 = [[x1, x2, x3, ...],[y1, y2, y3, ...]]
    # for each and all of our clusters, this is plotting all of the (x,y) points assiged to that cluster from our k_means function.
    for i, cluster in enumerate(clusters):
        x_coords = [point[0] for point in cluster]
        y_coords = [point[1] for point in cluster]
        plt.scatter(x_coords, y_coords, color=colors[i], label=f'Cluster {i+1}')

    # Then we do the same thing with plotting all of the centroids we calculated
    x_centroids = [centroid[0] for centroid in centroids]
    y_centroids = [centroid[1] for centroid in centroids]
    plt.scatter(x_centroids, y_centroids, color='k', marker='x', s=100, label='Centroids')

    # just graph formatting
    plt.title('K-Means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

####################################
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'Data set 1.txt')
x_coords, y_coords = x_y(file_path)
points = list(zip(x_coords, y_coords))
####################################
#            USER INPUT            #
####################################
## user enters their desired number of clusters and which initilization algorithm they want to use:
k = int(input("Enter the desired number of clusters (K): "))
selection = input("Enter selection criteria (R for random, K for K++ initial centroid collection): ")
## this is just some error handling
if selection != 'R' and selection != 'K':
        print("Invalid selection criteria.")
        exit()
## user enters their desired termination criteria and the corresponding value:
termination_criteria = input("Enter termination criteria (1 for number of iterations, 2 for SSE improvement): ")

## depending on which criteria they chose above, the program will either ask them for a maximum number of iterations
## or a desired SSE improvement percentage. We also start the run time here.
if termination_criteria == '1':
    max_iterations = int(input("Enter the maximum number of iterations: "))
    start_time = time.time()
    centroids, clusters, sse, iteration = k_means(points, k, termination_criteria, 1, max_iterations, selection)
elif termination_criteria == '2':
    sse_improvement = float(input("Enter the desired SSE improvement percentage (e.g., 1 for 1%): ")) / 100
    start_time = time.time()
    centroids, clusters, sse, iteration = k_means(points, k, termination_criteria, sse_improvement, 0, selection)
else:
    print("Invalid termination criteria.")
    exit()

## We print our results of the sse we got, how many iterations it took, and the run time.
print("\n")
print(f"Final sum of squared error (SSE): {sse} found in {iteration} iterations.")
print("Runtime of: --- %s seconds ---" % (time.time() - start_time))
## This then calls a function to display/create the graph of our centroids and clusters
visualize_clusters(centroids, clusters)
