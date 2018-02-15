# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:13:59 2018

@author: JoanWang

Python version: 3.5.2
"""
import sys
import pandas as pd
import numpy as np
from time import time
from itertools import repeat
import multiprocessing as mp
from scipy.spatial.distance import cdist


def find_nearest_centroid(arr, centroids, current_assign):
    '''
    For each data point in arr, find centroid that is shortest Euclidean distance away 
    Also calculate number of reassignments when compared to previous assignment of centroids

    Inputs
        arr (array): array of data points; a portion of original dataset
        centroids (array): array of dimension num_k x num_cols indicating new locations of centroids
        current_assign (array): 1-D array indicating assignment of each data point to a centroid
            length equals number of data points in arr
            
    Outputs
        new_assign (array): 1-D array indicating assignment of each data point to a centroid
            length equals number of data points in arr
        num_changes (int): number of points in arr that got reassigned to a different centroid
    '''
    # calculate distances between each pair of rows from each object
    # distances has dimensions (num_rows in arr) x (num_rows in centroids)
    distances = cdist(arr, centroids)

    # new centroid for each point is index of minimum distance
    new_assign = np.argmin(distances, axis = 1)

    # compute number of assignment changes that occurred
    num_reassign = sum(current_assign != new_assign)
    
    return new_assign, num_reassign

def calculate_centroids(data, all_assign, num_k):
    '''
    Recalculate the new centroids (means) for given assignment of data points to centroids
    
    Inputs
        data (array): array of data points 
        all_assign (array): 1-D array indicating assignment of each data point to a centroid
            length equals number of data points
        num_k (int): number of i's for k-means algorithm

    Outputs
        new_centroids (array): array of dimension num_k x num_cols indicating new locations of centroids
    '''
    new_centroids = []
    for k in range(num_k):
        points = data[all_assign == k]
        new_centroids.append(np.mean(points, axis = 0, keepdims = True))

    return np.concatenate(new_centroids)

    
def run_kmeans(filename, num_k, num_proc, max_iter):
    '''
    Function to run k-means algorithm on given file

    Inputs
        filename (str): name of csv or text file containing raw data
        num_k (int): number of k's for k-means algorithm
        num_proc (int): number of processors to use; cannot exceed machine's max processors
        max_iter (int): maximum number of iterations to allow before stopping

    Outputs:  
        all_assign (array): 1-D array indicating assignment of each data point to a centroid
            length equals number of data points in filename
        centroids (array): array of dimension num_k x num_cols indicating final locations of centroids
    '''
    # set start time
    stime = time()

    # confirm that num_proc does not exceed cpu_count of current machine
    num_cpus = mp.cpu_count()
    if num_proc > num_cpus:
        raise AssertionError("num_processors too large -- max allowed is {}".format(num_cpus))
    
    # read in csv as numpy array; ignore the first row  of headers and first column of indices
    data = np.genfromtxt(filename, delimiter = ',', skip_header = True)
    data = data[:, 1:]
        
    # randomly sample num_k observations from data to use as initial centroids
    # setting seed of random number generator at 0 for reproducibility
    np.random.seed(0)
    indices = np.random.randint(0, data.shape[0], num_k)
    centroids = data[indices]
    
    # split dataset into num_proc pieces to parallelize the process of finding the nearest centroid
    splits = np.array_split(data, num_proc)
    
    # initialize assignments of points to centroids as an empty array
    assign = [np.empty(s.shape[0]) for s in splits]

    # define arguments list
    args = zip(splits, repeat(centroids), assign)

    # initialize variables. num_changes is initialized to num_rows (i.e. a large number)
    num_rows = data.shape[0]
    num_reassign = num_rows
    num_iters = 0 

    # stopping conditions for while loop: when max_iter is exceeded OR fewer than 0.1% of points
    # get reassigned in a cycle
    while num_reassign >= 0.001*num_rows and num_iters <= max_iter:
        
        with mp.Pool(processes=num_proc) as pool:
            results = pool.starmap(find_nearest_centroid, args)

        # extract new assignments from results
        assign = [x[0] for x in results]
        all_assign = np.concatenate(assign)
        
        # calculat total number of reassignments that happened
        num_reassign = sum([x[1] for x in results])

        # recalculate centroids
        centroids = calculate_centroids(data, all_assign, num_k)

        print('Iteration {}: {} reassignments'.format(num_iters, num_reassign))

        # set up arguments for next cycle; splits for parallelization stay the same each time
        num_iters += 1
        args = zip(splits, repeat(centroids), assign)
      
    print("\nTotal time: ", time()-stime)

    return all_assign, centroids


if __name__ == '__main__':
    
    if len(sys.argv) == 4:
        num_k = sys.argv[1]
        num_proc = sys.argv[2]
        max_iter = sys.argv[3]
        all_assign, centroids = run_kmeans('County_Mortgage_Funding.csv', int(num_k), int(num_proc), int(max_iter))
    else:
        s = "usage: python3 {0} num_k num_proc max_iter"
        s = s.format(sys.argv[0])
        print(s)
        sys.exit(0)

    