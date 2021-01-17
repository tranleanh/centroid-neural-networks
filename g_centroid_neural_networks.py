#================================================================
#
#   Copyright (C) 2021 Tran Le Anh (Lean Tran)
#
#   Editor      : Sublime Text
#   Application : G - Centroid Neural Networks
#   Author      : tranleanh
#   Created date: 2021-01-16 02:20
#   Description : Public
#   Version     : 1.0
#
#================================================================

import numpy as np
from matplotlib import pyplot as plt


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
def centroid_neural_network(X, n_clusters=10, max_iteration = 100, epsilon=0.05):
    
    '''
        Centroid Neural Networks:
            X: input data
            n_clusters: num of clusters
            max_iteration
            
        Variables:
            centroids: final centroids
            w: final weights
            cluster_indices: len = len(data), labels
            cluster_elements: len = len(centroids), to store elements in each cluster     
    '''
    
    centroid_X = np.average(X[:, -len(X[0]):], axis=0)
    epsilon = 0.05

    w1 = centroid_X + epsilon
    w2 = centroid_X - epsilon

    w = []
    w.append(w1)
    w.append(w2)
    
    
    ########## EPOCH 0 ##########
    initial_clusters = 2

    # Create an array to store elements in each cluster
    cluster_elements = []     
    for cluster in range(initial_clusters):
        cluster_i = []
        cluster_elements.append(cluster_i)

    cluster_lengths = np.zeros(initial_clusters, dtype=int)

    # Create an array to label for each element
    cluster_indices = []

    for i in range(len(X)):
        x = X[i]

        distances = []
        for w_i in w:
            dist = np.linalg.norm(x-w_i)
            distances.append(dist)

        # find winner neuron
        index = np.argmin(distances)

        # add cluster index of data x to a list
        cluster_indices.append(index)

        # update winner neuron
        w[index] = w[index] + 1/(1+cluster_lengths[index])*(x - w[index])

        # append data to cluster
        cluster_elements[index].append(x)

        cluster_lengths[index] += 1

    centroids = []
    for elements in cluster_elements:
        elements = np.array(elements)
        centroid_i = np.average(elements[:, -len(elements[0]):], axis=0)
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
                dist = np.linalg.norm(x-w_i)
                distances.append(dist)

            # find winner neuron of x
            current_cluster_index = np.argmin(distances)

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

        centroids = []
        for elements in cluster_elements:
            elements = np.array(elements)
            centroid_i = np.average(elements[:, -len(elements[0]):], axis=0)
            centroids.append(centroid_i)

        print(epoch+1)

        if loser == 0: 
            if len(w) == num_of_all_clusters:
                print("Reach the Desired Number of Clusters. Stop at Epoch ", epoch+1)
                break

            else:
                all_error = []
                for i in range(len(centroids)):

                    # calculate error
                    error = 0
                    for x in cluster_elements[i]:

                        dist_e = np.linalg.norm(x-centroids[i])
                        error += dist_e

                    all_error.append(error)

                splitted_index = np.argmax(all_error)

                new_w = w[splitted_index] + epsilon
                w.append(new_w)

                new_cluster_thing = []
                new_cluster_thing = np.array(new_cluster_thing)

                cluster_elements.append(new_cluster_thing)

                cluster_lengths = list(cluster_lengths)
                cluster_lengths.append(0)
                cluster_lengths = np.array(cluster_lengths)
                
    return centroids, w, cluster_indices, cluster_elements


# Centroid Neural Networks with Detected Weights
def centroid_neural_network_detected_weights(input_data, detected_weights, n_clusters, epochs = 10):
    X = input_data
    w = detected_weights
    initial_clusters = len(w)


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
            dist = np.linalg.norm(x-w_i)
            distances.append(dist)

        # find winner neuron
        index = np.argmin(distances)

        # add cluster index of data x to a list
        cluster_indices.append(index)

        # update winner neuron
        w[index] = w[index] + 1/(1+cluster_lengths[index])*(x - w[index])

        # append data to cluster
        cluster_elements[index].append(x)
        cluster_lengths[index] += 1

    centroids = []
    for elements in cluster_elements:
        elements = np.array(elements)
        centroid_i = np.average(elements[:, -len(elements[0]):], axis=0)
        centroids.append(centroid_i)

    for epoch in range(epochs):
        loser = 0

        for i in range(len(X)):
            x = X[i]

            distances = []
            for w_i in w:
                dist = np.linalg.norm(x-w_i)
                distances.append(dist)

            # find winner neuron of x
            current_cluster_index = np.argmin(distances)

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

        centroids = []
        for elements in cluster_elements:
            elements = np.array(elements)
            centroid_i = np.average(elements[:, -len(elements[0]):], axis=0)
            centroids.append(centroid_i)

        print(epoch+1, len(centroids))

        if loser == 0: 
            if len(w) == n_clusters:
                print("Reach the Desired Number of Clusters. Stop at Epoch ", epoch+1)
                break

            else:
                all_error = []
                for i in range(len(centroids)):

                    # calculate error
                    error = 0
                    for x in cluster_elements[i]:

                        dist_e = np.linalg.norm(x-centroids[i])
                        error += dist_e

                    all_error.append(error)

                splitted_index = np.argmax(all_error)

                new_w = w[splitted_index] + epsilon
                w.append(new_w)

                new_cluster_thing = []
                new_cluster_thing = np.array(new_cluster_thing)

                cluster_elements.append(new_cluster_thing)

                cluster_lengths = list(cluster_lengths)
                cluster_lengths.append(0)
                cluster_lengths = np.array(cluster_lengths)

    return centroids, w, cluster_indices, cluster_elements


# G-CNN
def g_centroid_neural_network(input_data, num_clusters, num_subdata = 10, max_iteration = 50, epsilon = 0.05):

    X = input_data
    new_data = []
    for i in range(num_subdata):
        subdata = []
        for j in range(len(X)//num_subdata):
            x_i = X[(len(X)//num_subdata)*i + j]
            subdata.append(x_i)
        new_data.append(subdata)
    new_data = np.array(new_data)
    # print(np.array(new_data).shape)

    centroids = []
    w = []
    cluster_indices = []
    cluster_elements = []

    for i in range(len(new_data)):
        subdata_i = new_data[i]

        centroids_, w_, cluster_indices_, cluster_elements_ = centroid_neural_network(subdata_i, num_clusters, max_iteration, epsilon)

        centroids.append(centroids_)
        w.append(w_)
        cluster_indices.append(cluster_indices_)
        cluster_elements.append(cluster_elements_)

    # Create New Data with Detected Centroids
    gen2_data = []
    for centroids_i in centroids:
        for centroid_ii in centroids_i: 
            gen2_data.append(centroid_ii)

    gen2_data = np.array(gen2_data)

    # Run G-CNN one more time
    centroids_2, w_2, cluster_indices_2, cluster_elements_2 = centroid_neural_network(gen2_data, num_clusters, max_iteration, epsilon)

    # Run G-CNN last time
    detected_weights = centroids_2
    centroids, weights, cluster_indices, cluster_elements = centroid_neural_network_detected_weights(X, detected_weights, num_clusters, max_iteration)
    print("Reach the Desired Number of Clusters. Stop!")
    
    return centroids, weights, cluster_indices, cluster_elements



# G-CNN v2
def g_centroid_neural_network_2(input_data, num_clusters, num_subdata = 10, max_iteration = 50, epsilon = 0.05):

    X = input_data
    new_data = []
    for i in range(num_subdata):
        subdata = []
        for j in range(len(X)//num_subdata):
            x_i = X[(len(X)//num_subdata)*i + j]
            subdata.append(x_i)
        new_data.append(subdata)
    new_data = np.array(new_data)
    # print(np.array(new_data).shape)

    centroids = []
    w = []
    cluster_indices = []
    cluster_elements = []

    for i in range(len(new_data)):
        subdata_i = new_data[i]

        if i == 0:
            centroids_, w_, cluster_indices_, cluster_elements_ = centroid_neural_network(subdata_i, num_clusters, max_iteration, epsilon)

        else:
            detected_weights = w[0]
            centroids_, w_, cluster_indices_, cluster_elements_ = centroid_neural_network_detected_weights(subdata_i, detected_weights, num_clusters,max_iteration)

        centroids.append(centroids_)
        w.append(w_)
        cluster_indices.append(cluster_indices_)
        cluster_elements.append(cluster_elements_)

    # Create New Data with Detected Centroids
    gen2_data = []
    for centroids_i in centroids:
        for centroid_ii in centroids_i: 
            gen2_data.append(centroid_ii)

    gen2_data = np.array(gen2_data)

    centroids_2, w_2, cluster_indices_2, cluster_elements_2 = centroid_neural_network(gen2_data, num_clusters, max_iteration, epsilon)

    # Run G-CNN one more time
    detected_weights = centroids_2
    centroids, weights, cluster_indices, cluster_elements = centroid_neural_network_detected_weights(X, detected_weights, num_clusters, max_iteration)
    print("Reach the Desired Number of Clusters. Stop!")
    
    return centroids, weights, cluster_indices, cluster_elements


def plot_cnn_result(input_data, centroids, cluster_indices, figure_size=(8,8)):

    X = input_data
    num_clusters = len(centroids)

    plt.figure(figsize=figure_size)

    cnn_cluster_elements = []

    for i in range(num_clusters):
        display = []
        for x_th in range(len(X)):
            if cluster_indices[x_th] == i:
                display.append(X[x_th])

        cnn_cluster_elements.append(display)

        display = np.array(display)
        plt.scatter(display[:,0], display[:,1])
        plt.scatter(centroids[i][0], centroids[i][1], s=200, c='red')
        plt.text(centroids[i][0], centroids[i][1], f"Cluster {i}", fontsize=14)        

    plt.show()