"# K-Means-IE599" 
From scratch implementation of K-Means, with two different itilization strategies(Random, K-means++)
User will be prompted to enter three fields:
K, Selection, and Termination Criteria:
K = # of clusters
Selection = Random or K++ initilization('R' for random, 'K' for K++)
Termination Criteria = After a set # of iterations or after a upper bound improvement percentage is met.('1' for number of iterations, '2' for SSE improvement)

If you select 2, for SSE improvement you can set it equal to 0 for a guaranteed local minima of Sum of Squared Errors
