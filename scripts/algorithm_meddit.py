import collections
import matplotlib, datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp_sparse
import pickle, time
from multiprocessing import Pool
import cProfile, os, argparse
from scipy.sparse import hstack
import imp
import os,binascii, datetime
import multiprocessing as mp
import itertools
import logging, data_loader, helper
import sys
logging.basicConfig(level=logging.DEBUG,format='(%(threadName)-10s) %(message)s',)


"""
    Performs the initialization step.
    This is called only once.
    n_jobs > 1 will give errors
"""
def initialise(data, init_size, dist_func):
    n = data.shape[0]
    tmp_pos = np.array(np.random.choice(n, size=init_size, replace=False), dtype='int')
    estimate = np.mean(dist_func(data, data[tmp_pos]), axis = 1)
    return estimate

"""
    Main function
    exp_index   : A number to indicate the experiment number. Also acts as a seed
    data_loader : Function to load the data
    dataset_name: Name of the dataset. Used to save the results
    dist_func   : Function to evaluate the distances
    
"""
def Meddit(arg_tuple):
    exp_index   = arg_tuple[0]
    data_loader = arg_tuple[1]
    # data = arg_tuple[1]
    dataset_name= arg_tuple[2]
    dist_func   = arg_tuple[3]
    sigma       = arg_tuple[4]
    verbose     = arg_tuple[5]

    filename = '../experiments/NIPS2019sim/' + dataset_name + '/meddit/' + str(exp_index) + '.pkl'
    if os.path.isfile(filename):
        print("already did", filename)
        return -1


    np.random.seed(exp_index) #Random seed for reproducibility
    # Variable initialization
    data      = data_loader()
    n         = data.shape[0]

    num_init_pulls = 1 # can change
    Delta     = 1.0/n #Accuracy parameter. Increase this if you want to increase the speed
    num_arms  = 32 #Number of arms to be pulled in every round parallelly 32 originally
    step_size = 32 #Number of distance evaluation to be performed on every arm 32 originally
    lcb       = np.zeros(n, dtype='float')       #At any point, stores the mu - lower_confidence_interval
    ucb       = np.zeros(n, dtype='float')       #At any point, stores the mu + lower_confidence_interval
    T         = num_init_pulls*np.ones(n, dtype='int')#At any point, stores number of times each arm is pulled
    
    #Calculating the approximate std deviation
    # sample_distance = dist_func( data[np.random.randint(n,size=2000)], data[np.random.randint(n,size=2000)] ).flatten()
    # sigma = np.std(sample_distance)
    # print("sigma = ", sigma)

    # Bookkeeping variables
    start_time = time.time()
    summary    = []
    pullsarr =[]

        
    old_tmean  = 0
    """
        Chooses the "num_arms" arms with lowest lcb and removes the ones which have been pulled n times.
        Returns None at stopping time
    """
    def choose_arm():
        low_lcb_arms = np.argpartition(lcb,num_arms)[:num_arms]
        
        #Arms which are pulled >= ntimes and ucb!=lcb
        arms_pulled_morethan_n = low_lcb_arms[ np.where( (T[low_lcb_arms]>=n) & (ucb[low_lcb_arms] != lcb[low_lcb_arms]) ) ]
        
        if arms_pulled_morethan_n.shape[0]>0:
            # Compute the distance of these arms accurately
            estimate[arms_pulled_morethan_n] = np.mean(dist_func(data[arms_pulled_morethan_n], data), axis=1 )
            T[arms_pulled_morethan_n]  += n
            ucb[arms_pulled_morethan_n] = estimate[arms_pulled_morethan_n]
            lcb[arms_pulled_morethan_n] = estimate[arms_pulled_morethan_n]
        
        if ucb.min() <  lcb[np.argpartition(lcb,1)[1]]: #Exit condition
            return None

        arms_to_pull          = low_lcb_arms[ np.where(T[low_lcb_arms]<n) ]
        return arms_to_pull


    """
        Pulls the "num_arms" arms "step_size" times. Updates the estimate, ucb, lcb
    """
    def pull_arm(arms):
        # availableArms = list(set(range(n)) - set(arms))
        tmp_pos      = np.array( np.random.choice(n, size=step_size, replace=False), dtype='int')    
        X_arm        = data[arms]
        X_other_arms = data[tmp_pos]

        Tmean = np.mean(dist_func(X_arm,X_other_arms), axis=1)
        estimate[arms]   = (estimate[arms]*T[arms] + Tmean*step_size)/( T[arms] + step_size + 0.0 )
        T[arms]          = T[arms]+step_size

        lcb[arms]        = estimate[arms] - np.sqrt(sigma**2*np.log(1/Delta)/(T[arms]+0.0))
        ucb[arms]        = estimate[arms] + np.sqrt(sigma**2*np.log(1/Delta)/(T[arms]+0.0))

    
    #Step 1: Initialize
    print("running experiment ", exp_index, "with sigma", sigma)
    estimate = initialise(data, num_init_pulls, dist_func) ##Tavor Changed
    lcb      = estimate - np.sqrt(sigma**2*np.log(1/Delta)/num_init_pulls)
    ucb      = estimate + np.sqrt(sigma**2*np.log(1/Delta)/num_init_pulls)

    #Step 2: Iterate
    for ind in range(n*10):   
        #Choose the arms
        arms_to_pull = choose_arm()

        if ind%50 ==0:
            medoid = 17094
            print("for exp", exp_index, " for medoid ", lcb[medoid], np.min(lcb), estimate[medoid], np.min(estimate))
        
        #Stop if we have found the best arm
        if arms_to_pull is None:
            #Collecting final stats
            summary.append(estimate.argmin())
            pullsarr.append(T.mean())
            logging.info("exp {} done. Best arm = {}".format(exp_index, np.argmin(lcb)))
            print("Summary exp {}: avg pulls= {:.2F}, time = {:.2F}".format(exp_index, T.mean(), time.time()-start_time))
            break

        

        #Pull the arms
        pull_arm(arms_to_pull)

        left_over = np.where(lcb <= np.min(ucb))
        #Stats
        if ind%200 ==0:
            print("For {} after {} pulls {} is best".format(exp_index, T.mean(), estimate.argmin()))
        if ind%50 == 0:

            summary.append(estimate.argmin())
            pullsarr.append(T.mean())


        if T.mean() > old_tmean:
            old_tmean = T.mean() + 10
            thrown_away = (100.0*np.where(lcb > np.min(ucb))[0].shape[0])/n
            if verbose:
                logging.info(str(exp_index)+" Thrown away "+" "+str(thrown_away)+" "+str(T.mean())+" "+str(T.std()) ) 
                
    with open(filename,'wb') as f:
        pickle.dump([summary, pullsarr, time.time() - start_time, num_init_pulls, T.mean(), T] ,f)


ap = argparse.ArgumentParser(description="Reproduce the experiments in the manuscript")
ap.add_argument("--dataset",  help="Name of the dataset eg. rnaseq20k, netflix100k")
ap.add_argument("--num_exp",  help="max size of any split file.", type=int, default=32 )
ap.add_argument("--num_jobs", help="Num of parallel experiments", type=int, default=32 )
ap.add_argument("--verbose",  help="Running outputs", type=bool, default=False )

args = ap.parse_args()

num_jobs   = args.num_jobs
num_trials = args.num_exp
dataset    = args.dataset        
verbose    = args.verbose        

if dataset   == 'rnaseq20k':
    data_loader = data_loader.load_rnaseq20k
    dist_func   = helper.l1_dist
    sigma       = 0.25
elif dataset == 'rnaseq100k':
    data_loader = data_loader.load_rnaseq100k
    dist_func   = helper.l1_dist
    sigma       = 0.5
elif dataset == 'netflix20k':
    data_loader = data_loader.load_netflix20k
    dist_func   = helper.cosine_dist
    sigma       = 0.2
elif dataset == 'netflix100k':
    data_loader = data_loader.load_netflix100k
    dist_func   = helper.cosine_dist
    sigma       = 0.2
elif dataset == 'mnist':
    data_loader = data_loader.load_mnist
    dist_func   = helper.l2_dist
    sigma=-1
elif dataset == 'panT':
    data_loader = data_loader.load_10x_3k_panT
    dist_func = helper.l1_dist
    # sigma=215
    sigma=0.2

print("Running", num_trials, "experiments on ", num_jobs, "parallel jobs", "on dataset", dataset)
arg_tuple =  itertools.product(range(num_trials), [data_loader], [dataset], [dist_func], [sigma], [verbose] )

# if num_jobs == 1:
#     print("running serially")
#     for row in arg_tuple:
#         Meddit(row)
# else:
#     print("Running parallel with {} cores".format(num_jobs))
pool      = mp.Pool(processes=num_jobs)
pool.map(Meddit, arg_tuple)
