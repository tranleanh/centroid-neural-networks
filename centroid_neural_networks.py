#================================================================
#
#   Copyright (C) 2024 Tran Le Anh
#
#   Application : Centroid Neural Networks
#   Author      : tranleanh
#   Version     : 3.0.0
#
#================================================================

import numpy as np
from scipy.spatial.distance import cdist


def remove_element(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')



# Centroid Neural Networks
def centroid_neural_net(X, n_clusters, max_iteration = 100, epsilon = 0.05):

    centroid_X = np.mean(X, axis=0)

    w = [centroid_X + epsilon, centroid_X - epsilon]
    
    ########## EPOCH 0 ##########
    cluster_members = [[], []]

    cluster_indices = []

    for i, x in enumerate(X):

        distances = cdist([x], w ,'euclidean')[0]

        # find winner neuron
        index = np.argmin(distances)

        # add cluster index of data x to a list
        cluster_indices.append(index)

        # update winner neuron
        w[index] = w[index] + 1/(1+len(cluster_members[index]))*(x - w[index])

        # append data to cluster
        cluster_members[index].append(x)

        
    ########## EPOCH 1+ - INCREASE NUM OF CLUSERS ##########
    num_of_all_clusters = n_clusters

    for epoch in range(max_iteration):
        loser = 0

        for i in range(len(X)):
            x = X[i]

            distances = cdist([x], w ,'euclidean')[0]

            # find winner neuron of x
            current_cluster_index = np.argmin(distances)

            # what was the winner for x in previous epoch
            x_th = i
            previous_cluster_index = cluster_indices[x_th]

            # check if current neuron is a loser
            if previous_cluster_index != current_cluster_index: 
                # update winner neuron
                w[current_cluster_index] = w[current_cluster_index] + (x - w[current_cluster_index])/(len(cluster_members[current_cluster_index])+1)

                # update loser neuron
                w[previous_cluster_index] = w[previous_cluster_index] - (x - w[previous_cluster_index])/(len(cluster_members[previous_cluster_index])-1)

                # add and remove data to cluster    
                cluster_members[current_cluster_index] = list(cluster_members[current_cluster_index])
                cluster_members[current_cluster_index].append(x)
                remove_element(cluster_members[previous_cluster_index], x)  
    
                # update cluster index
                cluster_indices[x_th] = current_cluster_index

                loser += 1

        if loser == 0: 
            if len(w) == num_of_all_clusters:
                # print("Reach the Desired Number of Clusters. Stop at Epoch ", epoch+1)
                break

            else:
                all_error = []
                for i in range(len(w)):

                    dists = cdist([w[i]], cluster_members[i] ,'euclidean')[0]
                    error = np.sum(dists)
                    all_error.append(error)

                new_w = w[np.argmax(all_error)] + epsilon
                w.append(new_w)

                cluster_members.append(np.array([]))
    
    return np.array(w), cluster_indices
    


# Centroid Neural Networks with Initialized Weights
def centroid_neural_net_init_weights(X, init_weights, max_iteration = 100):

    w = init_weights
    initial_clusters = len(w)

    cluster_members = []
    for cluster in range(initial_clusters):
        cluster_i = []
        cluster_members.append(cluster_i)

    cluster_lengths = np.zeros(initial_clusters, dtype=int)
    cluster_indices = []

    for i in range(len(X)):
        x = X[i]

        distances = cdist([x], w ,'euclidean')[0]

        # find winner neuron
        index = np.argmin(distances)

        # add cluster index of data x to a list
        cluster_indices.append(index)

        # update winner neuron
        w[index] = w[index] + 1/(1+cluster_lengths[index])*(x - w[index])

        # append data to cluster
        cluster_members[index].append(x)

        cluster_lengths[index] += 1


    for epoch in range(max_iteration):
        loser = 0

        for i in range(len(X)):
            x = X[i]

            distances = cdist([x], w ,'euclidean')[0]

            # find winner neuron of x
            current_cluster_index = np.argmin(distances)

            # what was the winner for x in previous epoch
            x_th = i
            previous_cluster_index = cluster_indices[x_th]

            # check if current neuron is a loser
            if previous_cluster_index != current_cluster_index: 
                # update winner neuron
                w[current_cluster_index] = w[current_cluster_index] + (x - w[current_cluster_index])/(len(cluster_members[current_cluster_index])+1)

                # update loser neuron
                w[previous_cluster_index] = w[previous_cluster_index] - (x - w[previous_cluster_index])/(len(cluster_members[previous_cluster_index])-1)

                # add and remove data to cluster    
                cluster_members[current_cluster_index] = list(cluster_members[current_cluster_index])
                cluster_members[current_cluster_index].append(x)
                remove_element(cluster_members[previous_cluster_index], x)  
    
                # update cluster index
                cluster_indices[x_th] = current_cluster_index

                loser += 1

        if loser == 0: 
            print("Stop at Epoch ", epoch+1)
            break

    return np.array(w), cluster_indices