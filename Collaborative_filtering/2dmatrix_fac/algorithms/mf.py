import time
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils.extmath import randomized_svd


class MF():

    def __init__(self, Ri, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - Ri    : original user-location matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter

        - Rt    : modified test set user-location matrix
        - self.modifiedIndices = modifiedInd

        """

        
        self.Ri = Ri
        self.num_users, self.num_items = Ri.shape
        print("matrix dimensions: ",Ri.shape)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        print("Creating a test set\n")
        self.Rt = self.zero_out(self.Ri,5)
        self.calculate_sparsity(self.Rt)

    def calculate_sparsity(self,R):
        """
        Calculates sparsity of original matrix
        R is the matrix you want to calculate sparsity of
        """

        non_zero_elements = np.nonzero(R)
        nnz_count = len(non_zero_elements[0])
        print(nnz_count)
        total_elements = R.shape[0] * R.shape[1]
        sparsity = 1 - nnz_count/total_elements
        print(f"Sparsity Level: {sparsity:.2%} ")
        return sparsity
    

    def svd(self):
        """
        svd does SVD decomposition on self.R and returns the predicted full matrix
        """
        start = time.time()
        u, s, vh = randomized_svd(self.Rt, self.K, self.iterations)
        finish = time.time()
        totaltime = finish - start
        print(f"svd took {totaltime} seconds")
        S = np.zeros((u.shape[1], vh.shape[0]))
        np.fill_diagonal(S,s)
        predicted = np.dot(u,np.dot(S,vh))
        self.Rf = predicted

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
        self.samples = [
            (i, j, self.Rt[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
        ]
        
        print("samples created...\n")

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            print("Iteration", i)
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
        return training_process
    
    def run_sgd(self):

        self.rmse(self.Ri,self.Rt)

        #train matrix factorization
        print("\nTraining ...\n")
        self.train()
        Rf = self.full_matrix()
        self.Rf = Rf
        L = np.rint(Rf)
        print("\nDone\n")

        print("Original=\n",self.Ri)
        print("Learnt=\n",self.Rf)

        return self.Rf
    
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
    
    def zero_out(self, Ri, n):
        """
        For each row in the matrix, set up to 'max_zeros' non-zero entries to zero.
        Assumes that each row has greater than 5 non-zero values

        Parameters:
        - Ri: A 2D NumPy array with users as rows and locations as columns.
        - max_zeros: The number of entries to set to zero per row.
        
        Returns:
        - A modified matrix with specified entries set to zero.
        """

        modified_matrix = np.copy(Ri)
        rows = modified_matrix.shape[0]
        modifiedInd = []
        
        for row_index in range(rows):
            nnzIndex = np.nonzero(modified_matrix[row_index])[0]
            if len(nnzIndex)>n:
                chosenInd = np.random.choice(nnzIndex,n,False)
                for ind in chosenInd:
                    modified_matrix[row_index,ind] = 0
                    modifiedInd.append((row_index,ind))

        self.modifiedIndices = modifiedInd
        return modified_matrix
    
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
    
    def rmse(self,R,R1):
        """
        Calculate the rmse of two matrices R and R1
        Returns RMSE between R and R1
        """

        diff = R-R1
        sq_diff = diff**2
        total_diff = np.sum(sq_diff)
        total_elements = R1.shape[0]*R1.shape[1]
        mse = total_diff/total_elements
        rmse = np.sqrt(mse)
        return rmse
    
    def rmse_modifiedElements(self,R,R1):
        """
        Calculate RMSE of two matrices R and R1 but only for modifiedIndices to see if we have been able to predict
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
        return rmse

    def evaluation(self):
        """
        Ri is the initial matrix
        Rf is the estimated matrix calculated from test matrix
        Calculates RMSE 
        """

        print("\nFinding Error on test set...\n")
 
        #indices of modified zero values in test set
        print("Initial RMSE on replaced values:", self.rmse_modifiedElements(self.Ri,self.Rt))
        print("Final RMSE on replaced values:", self.rmse_modifiedElements(self.Ri,self.Rf))
