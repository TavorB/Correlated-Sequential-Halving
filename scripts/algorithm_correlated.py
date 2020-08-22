import numpy as np
import pickle, time
from multiprocessing import Pool
import os, argparse
import multiprocessing as mp
import itertools
import logging, data_loader, helper
import sys
logging.basicConfig(level=logging.DEBUG,format='(%(threadName)-10s) %(message)s',)



"""
    Main function
    exp_index   : A number to indicate the experiment number. Also acts as a seed
    data_loader : Function to load the data
    dataset_name: Name of the dataset. Used to save the results
    dist_func   : Function to evaluate the distances
    
"""
def Meddit(arg_tuple):

    exp_index   = arg_tuple[0]
    arm_budget  = arg_tuple[1]
    data_loader = arg_tuple[2]
    dataset_name= arg_tuple[3]
    dist_func   = arg_tuple[4]
    verbose     = arg_tuple[5]


    filenamebase = '../experiments/NIPS2019sim/' + dataset_name +'/correlated_SH/'
    filename = filenamebase+str(arm_budget) + "_exp" + str(exp_index)+'.pkl'
    if os.path.isfile(filename):
        print("already did", filename)
        return -1
    
    if not os.path.exists(filenamebase):
        os.makedirs(filenamebase)

    
    np.random.seed(exp_index) #Random seed for reproducibility
    print("loading dataset")

    # Variable initialization
    data      = data_loader()
    n         = data.shape[0]

    num_init_pulls = 0
    T         = np.zeros(n, dtype='int')#At any point, stores number of times each arm is pulled

    T_budget = arm_budget*n


    # Bookkeeping variables
    start_time = time.time()
    summary = [0]
    summary_pulls = [0]
    estimate = np.zeros(n)
    old_tmean  = 0
    num_rounds = int(np.ceil(np.log2(n)))
    S = np.array(range(n))


    """
        Pulls the given arms num_pulls_per_arm times. Updates the estimate
    """
    def pull_arm(arms, num_pulls_per_arm):
        # print arms
        tmp_pos      = np.array( np.random.choice(n, size=num_pulls_per_arm, replace=False), dtype='int')    
        X_arm        = data[arms]
        X_other_arms = data[tmp_pos]

        Tmean = np.mean(dist_func(X_arm,X_other_arms), axis=1)
        estimate[arms]   = (estimate[arms]*T[arms] + Tmean*num_pulls_per_arm)/( T[arms] + num_pulls_per_arm + 0.0 )
        # estimate[arms] = Tmean # theoretically don't use past
        if num_pulls_per_arm == n:
            estimate[arms] = Tmean
        T[arms]          = T[arms]+num_pulls_per_arm
    
    print("running experiment{} with budget {}".format(exp_index, arm_budget))
    
    round_iter = 0
    while len(S) > 1:
        num_pulls_per_arm = int(min(n,np.floor(T_budget*1.0/len(S)/num_rounds)))
        if num_pulls_per_arm ==0:
            num_pulls_per_arm = 1

        
        pull_arm(S, num_pulls_per_arm)

        lst = estimate[S]
        summary.append(S[np.argmin(lst)])
        summary_pulls.append(T.mean())

        med = np.median(lst)
        locs = np.where(lst<=med)[0]

        if num_pulls_per_arm == n: # calculated exactly
            S = [S[np.argmin(lst)]]
            break


        print("Round {}, {} arms: {} pulls, cutoff was {:.3F}".format(round_iter,len(S),num_pulls_per_arm, med))
        S = S[locs]

        if len(S)==1:
            break
        round_iter+=1


    assert(len(S) == 1)
    total_time = time.time()-start_time
    logging.info("Exp {} done. Best arm = {}, in {:.2F} sec with {:.1F} avg pulls".format(exp_index, S[0], total_time, T.mean()))


    if os.path.isfile(filename):
        print('Already done')
    else:
        with open(filename,'wb') as f:
            pickle.dump([summary[-1],summary_pulls[-1], total_time, S[0]],f)


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
elif dataset == 'rnaseq100k':
    data_loader = data_loader.load_rnaseq100k
    dist_func   = helper.l1_dist
elif dataset == 'netflix20k':
    data_loader = data_loader.load_netflix20k
    dist_func   = helper.cosine_dist
elif dataset == 'netflix100k':
    data_loader = data_loader.load_netflix100k
    dist_func   = helper.cosine_dist
elif dataset == 'mnist':
    data_loader = data_loader.load_mnist
    dist_func   = helper.l2_dist



print("Running", num_trials, "experiments on ", num_jobs, "parallel jobs", "on dataset", dataset)
## What budgets you want to use (how many distance evaluations per point on average)
# valRange = np.arange(15,22,2) ## Example usage
valRange = [15]

arg_tuple =  itertools.product(range(num_trials), valRange, [data_loader], [dataset], [dist_func], [verbose] )

pool      = mp.Pool(processes=num_jobs)
pool.map(Meddit, arg_tuple)
