#From https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b

import numpy
import pickle

with open('matrix_data.pkl', 'rb') as file:
    loaded_matrix = pickle.load(file)

print(loaded_matrix)

def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter'''

    Q = Q.T

    for step in range(steps):
        print(step)
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])


        eR = numpy.dot(P,Q)

        e = 0

        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        # 0.001: local minimum
        if e < 0.001:

            break

    return P, Q.T


"""R = [

     [5,3,0,1],

     [4,0,0,1],

     [1,1,0,5],

     [1,0,0,4],

     [0,1,5,4],
    
     [2,1,3,0],

    ]"""

R = loaded_matrix

# N: num of User
N = len(R)
# M: num of Locations
M = len(R[0])
# Num of Features
K = 20

 
#randomize P and Q variables
P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)

 
nP, nQ = matrix_factorization(R, P, Q, K)

print(f"nP: {nP}")
print(f"nQ: {nQ}")

with open('nPmatrix20_data.pkl', 'wb') as file:
    pickle.dump(nP, file)

with open('nQmatrix20_data.pkl', 'wb') as file:
    pickle.dump(nQ, file)

nR = numpy.dot(nP, nQ.T)

print(nR)
