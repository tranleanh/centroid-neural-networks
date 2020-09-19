#================================================================
#
#   Copyright (C) 2020 Tran Le Anh
#
#   Editor      : Sublime Text
#   Application : Centroid Neural Network
#   Author      : tranleanh
#   Created date: 2020-09-19 18:00
#   Description : Public
#   Version     : 1.0
#
#================================================================

from matplotlib import pyplot as plt
import numpy as np

def remove_element(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

def centroid_neural_net(input_data, n_clusters, max_iteration = 10, epsilon = 0.05):
    X = input_data
    centroid_X = (np.average(X[:,0]), np.average(X[:,1]))
    
    w1 = [centroid_X[0] + epsilon, centroid_X[1] + epsilon]
    w2 = [centroid_X[0] - epsilon, centroid_X[1] - epsilon]

    w = []
    w.append(w1)
    w.append(w2)
    
    ########## EPOCH 0 ##########

    initial_clusters = 2

    cluster_elements = []
    for cluster in range(initial_clusters):
        cluster_i = []
        cluster_elements.append(cluster_i)

    cluster_lengths = np.zeros(initial_clusters, dtype=int)

    cluster_indices = []

    for i in range(len(X)):
        x = X[i]

        distances = []
        for w_i in w:
            dist = (x[0]-w_i[0])**2 + (x[1]-w_i[1])**2
            distances.append(dist)

        # find winner neuron
        index = distances.index(min(distances))

        # add cluster index of data x to a list
        cluster_indices.append(index)

        # update winner neuron
        w[index] = w[index] + 1/(1+cluster_lengths[index])*(x - w[index])

        # append data to cluster
        cluster_elements[index].append(x)
        # print(cluster_elements)

        cluster_lengths[index] += 1

    # cluster_elements = np.array(cluster_elements)  

    centroids = []
    for elements in cluster_elements:
        elements = np.array(elements)
        centroid_i = (np.average(elements[:,0]), np.average(elements[:,1]))
        centroids.append(centroid_i)
        
    ########## EPOCH 1+ - INCREASE NUM OF CLUSERS ##########

    num_of_all_clusters = n_clusters
    epochs = max_iteration

    for epoch in range(epochs):
        loser = 0

        for i in range(len(X)):
            x = X[i]

            distances = []
            for w_i in w:
                dist = (x[0]-w_i[0])**2 + (x[1]-w_i[1])**2
                distances.append(dist)

            # find winner neuron of x
            current_cluster_index = distances.index(min(distances))

            # what was the winner for x in previous epoch
            x_th = i
            previous_cluster_index = cluster_indices[x_th]

            # check if current neuron is a loser
            if previous_cluster_index != current_cluster_index:
                # update winner neuron
                w[current_cluster_index] = w[current_cluster_index] + (x - w[current_cluster_index])/(cluster_lengths[current_cluster_index]+1)

                # update loser neuron
                w[previous_cluster_index] = w[previous_cluster_index] - (x - w[previous_cluster_index])/(cluster_lengths[previous_cluster_index]-1)

                # add and remove data to cluster    
                cluster_elements[current_cluster_index] = list(cluster_elements[current_cluster_index])
                cluster_elements[current_cluster_index].append(x)
                remove_element(cluster_elements[previous_cluster_index], x)  
    
                # update cluster index
                cluster_indices[x_th] = current_cluster_index

                cluster_lengths[current_cluster_index] += 1
                cluster_lengths[previous_cluster_index] -= 1

                loser += 1

        # cluster_elements = np.array(cluster_elements)

        centroids = []
        for elements in cluster_elements:
            elements = np.array(elements)
            centroid_i = [np.average(elements[:,0]), np.average(elements[:,1])]
            centroids.append(centroid_i)

        if loser == 0: 
            if len(w) == num_of_all_clusters:
                # print("Loser = 0, reach the desired num of clusters")
                print("Reach the Desired Number of Clusters. Stop at Epoch ", epoch+1)
                break

            else:
                # print("Loser = 0, Now starting to split weight")
                all_error = []
                for i in range(len(w)):

                    # calculate error
                    error = 0
                    for x in cluster_elements[i]:
                        error += np.sqrt((x[0] - w[i][0])**2 + (x[1] - w[i][1])**2)

                    all_error.append(error)

                splitted_index = all_error.index(max(all_error))
                # print(f"Start to split Cluster {splitted_index}")

                new_w = [w[splitted_index][0] + epsilon, w[splitted_index][1] + epsilon]
                w.append(new_w)

                new_cluster_thing = []
                new_cluster_thing = np.array(new_cluster_thing)

                # cluster_elements = list(cluster_elements)
                cluster_elements.append(new_cluster_thing)
                # cluster_elements = np.array(cluster_elements)

                cluster_lengths = list(cluster_lengths)
                cluster_lengths.append(0)
                cluster_lengths = np.array(cluster_lengths)
    
    return centroids, w, cluster_indices, cluster_elements, cluster_lengths
