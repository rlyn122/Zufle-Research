import numpy as np
from sklearn.metrics import mean_squared_error
from cvxpy import *
from sklearn.utils.extmath import randomized_svd


class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        print("Creating a test set\n")
        self.Rt = MF.zero_out(self,5)


    def calculate_sparsity(self):
        """
        Calculates sparsity of matrix
        """
        total_elements = self.R.size
        non_zero_elements = np.count_nonzero(self.R)
        sparsity = 1 - non_zero_elements/total_elements
        return sparsity
    

    def svd(self):
        """
        svd does SVD decomposition on self.R and returns the predicted full matrix
        """

        u, s, vh = randomized_svd(self.Rt, n_components=self.K, n_iter=self.iterations)
        S = np.zeros((u.shape[1], vh.shape[0]))
        np.fill_diagonal(S,s)

        predicted = np.dot(u,np.dot(S,vh))
        return predicted

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Create a list of training samples
        self.samples = [
            (i, j, self.Rt[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
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
        modified_matrix = np.copy(self.R)
        rows = modified_matrix.shape[0]

        modifiedInd = []
        for row_index in range(rows):
            nonzero_indices = np.where(modified_matrix[row_index][:] != 0 )[0]
            five_nonzero_indices = np.random.choice(nonzero_indices,n)
            #delete 5 from each row
            modified_matrix[row_index][five_nonzero_indices] = 0
            modifiedInd.append((row_index,five_nonzero_indices))

        self.modifiedIndices = modifiedInd
        self.R_modified = modified_matrix
        self.R = modified_matrix
        return modified_matrix
    
    def evaluation(self,R,Rf):
        """
        R is the initial matrix
        Rf is the estimated matrix calculated from test matrix
        Calculates RMSE 
        """

        print("\nFinding Error on test set...\n")
 
        #indices of original non_zero values
        nonzeroIndices = np.transpose(np.nonzero(R))
        msef = 0
        for i,j in nonzeroIndices:
            msef += (R[i][j] - Rf[i][j])**2

        #indices of modified zero values in test set
        print("METHOD 1 : calculate error for only replaced values and non_zero")
        modifiedIndices = self.modifiedIndices
        for i,j in modifiedIndices:
            for jx in j:
                msef += (R[i][jx] - Rf[i][jx])**2
        msef /= len(nonzeroIndices)*len(nonzeroIndices[0])
        rmsef = msef**0.5

        print("RMSE final = ", rmsef)

        print("METHOD 2: calculate total error")
        msef = mean_squared_error(R,Rf)
        rmsef = msef**0.5

        print("RMSE final = ", rmsef)
