import pandas as pd
import numpy as np
import math
import re
import sklearn
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from surprise import Reader, Dataset
sns.set_style("darkgrid")
from cvxpy import *
from numpy import matrix
from sklearn.utils.extmath import randomized_svd


class MF():

    def __init__(self, Ri, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - Ri (csr)   : original user-location matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter

        - Rt (csr)    : modified test set user-location matrix
        - self.modifiedIndices = modifiedInd

        """

        
        self.Ri = csr_matrix(Ri)
        self.num_users, self.num_items = Ri.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

        #Create test set
        self.Rt = self.zero_out(5)


    def calculate_sparsity(self,R):
        """
        Calculates sparsity of original matrix
        R is the matrix you want to calculate sparsity of, should be in csr format
        """

        non_zero_elements = R.nnz
        print("non-zero elements:", non_zero_elements)
        total_elements = R.shape[0] * R.shape[1]
        sparsity = 1 - non_zero_elements/total_elements
        print(f"Sparsity Level: {sparsity:.2%} ")

        return sparsity
    

    def svd(self):
        """
        svd does SVD decomposition on self.R and returns the predicted full matrix
        """
        print("Original:\n",self.Ri)
        self.calculate_sparsity(self.Ri)

        print("Test Set:\n",self.Rt)
        self.calculate_sparsity(self.Rt)

        self.rmse(self.Ri,self.Rt)

        #train matrix factorization
        print("\nTraining ...\n")
        u,s,vh = randomized_svd(self.Ri,self.K,self.iterations)
        S = np.zeros((u.shape[1], vh.shape[0]))
        np.fill_diagonal(S,s)
        predicted = np.dot(u,np.dot(S,vh))
        self.Rf = csr_matrix(predicted)

        print("\nDone\n")

        print("Original=\n",self.Ri)
        print("Learnt=\n",self.Rt)

        return predicted

    def train(self):

        """
        Run SGD and create self.P and self.Q
        """

        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        # Create a list of training samples
        rows, cols = self.Rt.nonzero()
        self.samples = [(i,j,self.Rt[i,j]) for i,j in zip(rows,cols)]
        
        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            print("Iteration", i)
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i,mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        rows, cols = self.Rt.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(rows, cols):
            error += pow(self.Rt[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction =  self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using P and Q
        """
        return self.P.dot(self.Q.T)
    
    def zero_out(self, n):
        """
        For each row in the matrix, set up to 'max_zeros' non-zero entries to zero.
        Assumes that each row has greater than 5 non-zero values

        Parameters:
        - matrix: A 2D NumPy array with users as rows and locations as columns.
        - max_zeros: The number of entries to set to zero per row.
        
        Returns:
        - A modified matrix with specified entries set to zero.
        """

        #convert CSR to LIL for modificaiton
        modified_matrix = self.Ri.tolil()

        rows = modified_matrix.shape[0]
        modifiedInd = []
        
        #replace five indices with zero
        for row_index in range(rows):
            nonzero_indices = modified_matrix.rows[row_index]
            if (len(nonzero_indices)>5):
                indicestoZeroOut = np.random.choice(nonzero_indices,n,False)

                #set indices to zero
                for col_index in indicestoZeroOut:
                    modified_matrix[row_index,col_index] = 0 
                    modifiedInd.append((row_index,col_index))

        self.modifiedIndices = modifiedInd
        #convert matrix back to csr
        modified_matrix = modified_matrix.tocsr()

        return modified_matrix
    
    def rmse(self,R,R1):
        """
        Calculate the rmse of two csr matrices R and R1
        Returns RMSE between R and R1
        """

        diff = R-R1
        sq_diff = diff.power(2)
        total_diff = sq_diff.sum()

        total_elements = R1.shape[0]*R1.shape[1]
        mse = total_diff/total_elements
        rmse = np.sqrt(mse)
        print("RMSE = ",rmse)
        return rmse
    
    def rmse_modifiedElements(self,R,R1):
        """
        Calculate RMSE of two csr matrices R and R1 but only for modifiedIndices to see if we have been able to predict
        the modified indcies.

        Returns RMSE of modified elements of R and R1
        """
        error = 0
        count = 0
        modifiedIndices = self.modifiedIndices
        for i,j in modifiedIndices:
            error += (R[i,j]-R1[i,j])**2
            count+=1

        rmse = np.sqrt(error/count)
        print("RMSE on modified indices", rmse)
        return rmse
    
    def run_sgd(self):

        print("Original:\n",self.Ri)
        self.calculate_sparsity(self.Ri)

        print("Test Set:\n",self.Rt)
        self.calculate_sparsity(self.Rt)

        self.rmse(self.Ri,self.Rt)

        #train matrix factorization
        print("\nTraining ...\n")
        self.train()
        Rf = self.full_matrix()
        self.Rf = csr_matrix(Rf)
        L = np.rint(Rf)
        print("\nDone\n")

        print("Original=\n",self.Ri)
        print("Learnt=\n",self.Rt)

        return self.Rt

    def evaluation(self):
        """
        Ri is the initial matrix
        Rf is the estimated matrix calculated from test matrix
        Calculates RMSE 
        """

        print("\nFinding Error on test set...\n")
 
        #indices of modified zero values in test set
        print("METHOD 1 : calculate error for only replaced values")
        self.rmse_modifiedElements(self.Ri,self.Rf)

        print("METHOD 2: calculate total error")
        self.rmse(self.Ri,self.Rf)
