# Correlated-Sequential-Halving
## Finds the medoid of n points in O(nlog n) steps (naive approach takes O(n^2) steps).

This is a codebase to reproduce all the figures and numbers of the paper titled - "Ultra Fast Medoid Identification via Correlated Sequential Halving".

 1) All the figures can be viewed and generated via ipython notebooks in 'figure' folder
 2) The above figures are generated from experiments, which can be re-generated using the following code

  * python algorithm_rand.py --dataset *** --num_exp 1000 --num_jobs 32 --verbose False
  * python algorithm_brute.py --dataset *** --num_exp 1 --num_jobs 1 --verbose False
  * python algorithm_meddit.py --dataset *** --num_exp 1000 --num_jobs 32 --verbose False
  * python algorithm_correlated.py --dataset *** --num_exp 1000 --num_jobs 32 --verbose False

    * dataset - name of the dataset (rnaseq20k, rnaseq100k, netflix20k, netflix100k, mnist)
    * num_exp - Number of total experiments
    * num_jobs - Number of experiments run parallely
  
