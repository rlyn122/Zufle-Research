# Zufle-Resesarch

The goal of this research is to identify possible areas of optimization for location recommendation services. Using the Gowalla dataset from Stanford SNAP, I will use collaborative filtering and attempt to optimize location recommendation systems.

[Gowalla Dataset](https://snap.stanford.edu/data/loc-gowalla.html)

For reference, I am studying using [The Big Chaos Solution to the Netflix Grand Prize](https://www.asc.ohio-state.edu/statistics/statgen/joul_aut2009/BigChaos.pdf) as a case study.

https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/ProgressPrize2008_BellKor.pdf

The Netflix Prize was won by x`x   Team through a combination of Bellkor, Pragmatic Theory, and BigChaos teams' algorithms. The algorithms applied ranged from K-Nearest-Neighbors, Restricted-Boltzmann Machines, SVD++, Maximum Margin Matrix Factorization, and more. To combine predictors of multiple teams, a combination of nonlinear probe blending and linear quiz blending was applied.


Python 3.6

__Process__
1. Filter data within NY metropolitan area and users with 10+ checkins 
2. Import dataset and convert to nxm matrix with n users and m locations, where 'rating' values are amount of times checked in
2. Try to adapt Netflix Challenge Matrix Factorization for Location Recommendation

__Files__

1. mf.py 
2. evaluation.py
3. 


__Algorithms Explored:__ 
    
  - Collaborative Filtering/Matrix Factorization
      - Singular Value Decomposition (SVD)
      - 
  Learning Algorithms
    - Stochastic Gradient Descent (SGD)
    - Alternating Least Square (ALS)