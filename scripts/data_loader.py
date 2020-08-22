from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import scipy.sparse as sp_sparse
import tables, h5py
import pickle, time
import logging
import os,binascii, datetime
import helper
# from tensorflow.examples.tutorials.mnist import input_data

logging.basicConfig(level=logging.DEBUG,format='(%(threadName)-10s) %(message)s',)


def load_rnaseq20k():
    path        = '../datasets/rnaseq/1M_neurons_neuron20k.h5'
    gene_matrix = helper.get_matrix_from_h5(path, "mm10")
    data        = helper.normalise(gene_matrix)
    return data

def load_rnaseq100k():
    path        = '../datasets/rnaseq/cluster1/'
    gene_matrix = helper.get_matrix_from_h5_filepath(path, "mm10")
    data        = helper.normalise(gene_matrix)
    return data


def load_netflixdata():
    with h5py.File('../datasets/netflixdata.h5', 'r') as hf:
        data    = hf['data'][:]
        indices = hf['indices'][:]
        indptr  = hf['indptr'][:]
    data = sp_sparse.csc_matrix((data, indices, indptr)) #Sparse matrix
    return data 

def load_netflix20k():
    return load_netflixdata()[:20000]

def load_netflix100k():
    return load_netflixdata()[:100000]

def load_mnist():
    # mnist = input_data.read_data_sets("../datasets/MNIST/", one_hot=False)
    # data  = np.vstack([ mnist.train.images[np.where(mnist.train.labels==0)], mnist.test.images[np.where(mnist.test.labels==0)]])
    # return data
    print("Not supported")
    return -1