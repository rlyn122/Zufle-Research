import pickle
import numpy as np
from scipy.sparse import coo_matrix

def sgd_matrix_factorization_sparse(R, K, steps=100, alpha=0.0002, beta=0.02, batch_size=100):
    """
    Perform matrix factorization using Stochastic Gradient Descent (SGD) optimized for sparse matrices.

    :param R: The user-location rating sparse matrix
    :param K: Number of latent features
    :param steps: Number of iterations to perform
    :param alpha: Learning rate
    :param beta: Regularization parameter
    :param batch_size: Number of ratings to sample in each iteration
    :return: Final matrices P and Q
    """
    M, N = R.shape
    P = np.zeros((M, K))
    Q = np.zeros((N, K))
    Q = Q.T 
    
    for step in range(steps):
        print(f'Step: {step}')
        # Shuffle the indices to create random batches
        indices = np.arange(R.nnz)
        np.random.shuffle(indices)
        indices = indices[:batch_size]
        
        for idx in indices:
            i, j = R.row[idx], R.col[idx]
            rij = R.data[idx]
            
            eij = rij - np.dot(P[i, :], Q[:, j])
            for k in range(K):
                P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        
        e = sum(pow(R.data[idx] - np.dot(P[R.row[idx], :], Q[:, R.col[idx]]), 2) for idx in indices)
        print(f'Error: {e}')
        
    return P, Q.T


with open('../data/pkl/NYfiltered.pkl', 'rb') as file:
    R = pickle.load(file)

R_sparse = coo_matrix(R)
K = 10

nP, nQ = sgd_matrix_factorization_sparse(R_sparse, K)

# Evaluate the factorization
if nQ.shape[0] != nP.shape[1]:
    nQ = nQ.T    

predicted = np.dot(nP,nQ)

#Store as pkl file for easy access
with open("./output/sgd.pkl", 'wb') as file:
    pickle.dump(predicted, file)