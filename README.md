# Zufle-Resesarch

The goal of this research is to identify possible areas of optimization for location recommendation services. Using the Gowalla dataset from Stanford SNAP, I will use collaborative filtering and attempt to optimize location recommendation systems.

[Gowalla Dataset](https://snap.stanford.edu/data/loc-gowalla.html)

Python 3.6

__Process__
1. Filter data within NY metropolitan area and users with 10+ checkins 
2. Import dataset and convert to nxm matrix with n users and m locations, where 'rating' values are amount of times checked in
2. Apply Netflix Challenge Matrix Factorization for Location Recommendation following this [implementation.](https://github.com/harshraj11584/Paper-Implementation-Matrix-Factorization-Recommender-Systems-Netflix?tab=readme-ov-file#ieee-paper-matrix-factorization-techniques-for-recommender-systems)
3. 

__Files__

1. mf.py 
2. evaluation.py
3. 


__Algorithms Explored:__ 
    
  - Collaborative Filtering/Matrix Factorization
      - Singular Value Decomposition (SVD)
  Learning Algorithms
    - Stochastic Gradient Descent (SGD)
    - Alternating Least Square (ALS)