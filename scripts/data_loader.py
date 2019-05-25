from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import scipy.sparse as sp_sparse
import tables, h5py
import pickle, time
import logging
import os,binascii, datetime
import helper
# import scanpy as sc
# from tensorflow.examples.tutorials.mnist import input_data

logging.basicConfig(level=logging.DEBUG,format='(%(threadName)-10s) %(message)s',)


def load_birch1():
    with open('/data/medoid_bandit/Med-dit/datasets/birch1/birch1.txt', 'rb') as f:
        data = np.loadtxt(f)
    return data

def load_birch2():
    with open('/data/medoid_bandit/Med-dit/datasets/birch2/birch2.txt', 'rb') as f:
        data = np.loadtxt(f)
    return data

def load_birch3():
    with open('/data/medoid_bandit/Med-dit/datasets/birch3/birch3.txt', 'rb') as f:
        data = np.loadtxt(f)
    return data

def load_rnaseq20k():
    path        = '../datasets/rnaseq/1M_neurons_neuron20k.h5'
    # path = '/data/medoid_bandit/K-medoid-20k/data/rnaseq/1M_neurons_neuron20k.h5'
    gene_matrix = helper.get_matrix_from_h5(path, "mm10")
    data        = helper.normalise(gene_matrix)
    return data

def load_rnaseq100k():
    # path        = '../datasets/rnaseq/cluster1/'
    path        = '/data/medoid_bandit/Med-dit/datasets/rnaseq/cluster1/'
    gene_matrix = helper.get_matrix_from_h5_filepath(path, "mm10")
    data        = helper.normalise(gene_matrix)
    return data


def load_rand():
    with open('/data/tavorb/meddit/simGaussData.pkl', 'rb') as f:
        data = pickle.load(f, encoding = 'latin1')
    return data

def load_rand2():
    with open('/data/tavorb/meddit/simGaussData2.pkl', 'rb') as f:
        data = pickle.load(f, encoding = 'latin1')
    return data
    

def load_10x_3k_panT():
    """ 
        https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/t_3k
    """
    filename_data = '/data/martin/single_cell/10x_3k_panT/filtered_gene_bc_matrices/GRCh38/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_3k_panT/filtered_gene_bc_matrices/GRCh38/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_3k_panT/filtered_gene_bc_matrices/GRCh38/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    X= data.X.toarray()
    return helper.normalise(X.T)


def load_netflixdata():
    # path = '/data/medoid_bandit/Med-dit/datasets/netflix/'
    # # path = '/data/medoid_bandit/K-medoid-20k/data/netflix/'
    # # with h5py.File('../datasets/netflix/all_combined_data.h5', 'r') as hf:
    # # with h5py.File('/data/medoid_bandit/netflix/all_combined_data.h5', 'r') as hf:
    # with h5py.File(path +'all_combined_data.h5', 'r') as hf:
    #     data    = hf['data'][:]
    #     indices = hf['indices'][:]
    #     indptr  = hf['indptr'][:]
    # # data = sp_sparse.csc_matrix((data, indices, indptr)) #Sparse matrix
    # # return data 
    # return data, indices, indptr


    # path = '/data/medoid_bandit/K-medoid-20k/data/netflix/'
    # with h5py.File('../datasets/netflix/all_combined_data.h5', 'r') as hf:
    # with h5py.File('/data/medoid_bandit/netflix/all_combined_data.h5', 'r') as hf:
    with h5py.File('/data/tavorb/meddit/netflixdata_tavor.h5', 'r') as hf:
        data    = hf['data'][:]
        indices = hf['indices'][:]
        indptr  = hf['indptr'][:]
    data = sp_sparse.csc_matrix((data, indices, indptr)) #Sparse matrix
    return data 
    # return data, indices, indptr
    
    # datafile = open('/data/tavorb/meddit/netflixdata_tavor.pkl','rb')
    # data, indices, indptr = pickle.load(datafile)
    # return sp_sparse.csc_matrix((data, indices, indptr))

def load_netflix20k():
    return load_netflixdata()[:20000]

def load_netflix100k():
    return load_netflixdata()[:100000]

def load_mnist():
    mnist = input_data.read_data_sets("../../trimed_datasets/MNIST/", one_hot=False)
    data  = np.vstack([ mnist.train.images[np.where(mnist.train.labels==0)], mnist.test.images[np.where(mnist.test.labels==0)]])
    return data

def load_europe():
    #Returns europe in non-sparse format
    data = np.loadtxt('/data/medoid_bandit/E18_Mice/trimed_datasets/europe/europediff.txt')
    return data

def load_Gnutella_6k():
    filepath = '/data/medoid_bandit/E18_Mice/trimed_datasets/gnutella/dist_matrix_Gnutella_6k'
    with open(filepath,'rb') as f:
        data=pickle.load(f)
    return data

def load_Gnutella_6k_undirected():
    filepath = '/data/medoid_bandit/E18_Mice/trimed_datasets/gnutella/dist_matrix_Gnutella_6k_undirected.pkl' 
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# def load_MNIST(value):
#     from mlxtend.data import loadlocal_mnist
#     X, y = loadlocal_mnist(
#         images_path='/data/tavorb/meddit/mnist/train-images-idx3-ubyte', 
#         labels_path='/data/tavorb/meddit/mnist/train-labels-idx1-ubyte')

#     return X[np.where(y ==value)[0]].shape





def load_tiny_imagenet():
    return 0

# from scipy import ndimage, misc
# import imageio
#
# def load_tiny_imagenet():
#     d = 64*64*3
#     images = np.zeros((10000, d)) #n x d
#     ctr = 0
#     for root, dirnames, filenames in os.walk("/data/MAB/test_dataset/tiny-imagenet-200/val"):
#         for filename in filenames:
#             # if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
#             filepath = os.path.join(root, filename)
#             im = imageio.imread(filepath)
#             im = im.flatten()
#             if (im.size!= d):
#                 # print "grayscale"
#                 im = np.concatenate((im,im,im))
#             images[ctr,:] = im
#             ctr+=1
#             #flatten and add to np array
#
#             # image = ndimage.imread(filepath, mode="RGB")
#             # image_resized = misc.imresize(image, (64, 64))
#             # images.append(image_resized)
#     return images
# def load_tiny_imagenet(path, dtype=np.float32, val_only=True):
#     """
#     Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
#     TinyImageNet-200 have the same directory structure, so this can be used
#     to load any of them.

#     Inputs:
#     - path: String giving path to the directory to load.
#     - dtype: numpy datatype used to load the data.

#     Returns: A tuple of
#     - class_names: A list where class_names[i] is a list of strings giving the
#     WordNet names for class i in the loaded dataset.
#     - X_train: (N_tr, 3, 64, 64) array of training images
#     - y_train: (N_tr,) array of training labels
#     - X_val: (N_val, 3, 64, 64) array of validation images
#     - y_val: (N_val,) array of validation labels
#     - X_test: (N_test, 3, 64, 64) array of testing images.
#     - y_test: (N_test,) array of test labels; if test labels are not available
#     (such as in student code) then y_test will be None.
#     """
#   # First load wnids
#     with open(os.path.join(path, 'wnids.txt'), 'r') as f:
#         wnids = [x.strip() for x in f]

#   # Map wnids to integer labels
#     wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

#   # Use words.txt to get names for each class
#     with open(os.path.join(path, 'words.txt'), 'r') as f:
#         wnid_to_words = dict(line.split('\t') for line in f)
#         for wnid, words in wnid_to_words.iteritems():
#             wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
#     class_names = [wnid_to_words[wnid] for wnid in wnids]

#     # Next load training data.
#     X_train = []
#     y_train = []
#     img_files = os.listdir(os.path.join(path, 'test', 'images'))
#     X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
#     y_test = None

#     if not val_only:
#         for i, wnid in enumerate(wnids):
#             if (i + 1) % 20 == 0:
#                 print 'loading training data for synset %d / %d' % (i + 1, len(wnids))
#             # To figure out the filenames we need to open the boxes file
#             boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
#             with open(boxes_file, 'r') as f:
#                 filenames = [x.split('\t')[0] for x in f]
#             num_images = len(filenames)

#             X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
#             y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
#             for j, img_file in enumerate(filenames):
#                 img_file = os.path.join(path, 'train', wnid, 'images', img_file)
#                 img = imread(img_file)
#                 if img.ndim == 2:
#                     ## grayscale file
#                     img.shape = (64, 64, 1)
#                 X_train_block[j] = img.transpose(2, 0, 1)
#             X_train.append(X_train_block)
#             y_train.append(y_train_block)

#         # We need to concatenate all training data
#         X_train = np.concatenate(X_train, axis=0)
#         y_train = np.concatenate(y_train, axis=0)
        
#         # Next load test images
#         # Students won't have test labels, so we need to iterate over files in the
#         # images directory.
#         for i, img_file in enumerate(img_files):
#             img_file = os.path.join(path, 'test', 'images', img_file)
#             img = imread(img_file)
#             if img.ndim == 2:
#                 img.shape = (64, 64, 1)
#             X_test[i] = img.transpose(2, 0, 1)

#         y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
#         if os.path.isfile(y_test_file):
#             with open(y_test_file, 'r') as f:
#                 img_file_to_wnid = {}
#                 for line in f:
#                     line = line.split('\t')
#                     img_file_to_wnid[line[0]] = line[1]
#             y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
#             y_test = np.array(y_test)
  
#   # Next load validation data
#     with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
#         img_files = []
#         val_wnids = []
#         for line in f:
#             img_file, wnid = line.split('\t')[:2]
#             img_files.append(img_file)
#             val_wnids.append(wnid)
#         num_val = len(img_files)
#         y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
#         X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
#         for i, img_file in enumerate(img_files):
#             img_file = os.path.join(path, 'val', 'images', img_file)
#             img = imread(img_file)
#             if img.ndim == 2:
#                 img.shape = (64, 64, 1)
#             X_val[i] = img.transpose(2, 0, 1)
    
#     return class_names, X_train, y_train, X_val, y_val, X_test, y_test
