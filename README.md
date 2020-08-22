# Correlated-Sequential-Halving
## Finds the medoid of n points efficiently, approximately O(n log^2 n): brute force takes O(n^2) time.

This is a codebase to reproduce all the figures and numerical results in the paper titled - "Ultra Fast Medoid Identification via Correlated Sequential Halving". Please reach out to me at tavorb "at" stanford.edu if you have any questions on how to run the code or replicate results.

 1) All the figures can be viewed and generated via ipython notebooks in 'figure' folder
 2) The above figures are generated from experiments, which can be re-generated using the following code

  * python algorithm_rand.py --dataset *** --num_exp 1000 --num_jobs 32 --verbose False
  * python algorithm_brute.py --dataset *** --num_exp 1 --num_jobs 1 --verbose False
  * python algorithm_meddit.py --dataset *** --num_exp 1000 --num_jobs 32 --verbose False
  * python algorithm_correlated.py --dataset *** --num_exp 1000 --num_jobs 32 --verbose False
  	* budget can be modified by changing valRange in algorithm_correlated.py. Can perform 'doubling trick' to search and find a good budget.

With the following options:
* dataset - name of the dataset (rnaseq20k, netflix20k, netflix100k, mnist)
  * netflix dataset not included on github due to size, downloadable at https://www.kaggle.com/netflix-inc/netflix-prize-data
  * MNIST command deprecated
* num_exp - Number of total experiments
* num_jobs - Number of experiments run parallely
  
3) Dependencies: tables and h5py are used in loading rnaseq data, can be installed with python -m pip install h5py. Code is compatible with both Python 2 and 3.
