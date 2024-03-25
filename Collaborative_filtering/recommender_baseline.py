#Setting up prerequisites
#from numba import prange
from mf import MF
import pandas as pd
import numpy as np
import math
import re
import sklearn
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset
sns.set_style("darkgrid")
from cvxpy import *
from numpy import matrix
import pickle
print("Setup Complete\n")

def zero_out(matrix, n=5):
    """
    For each row in the matrix, set up to 'max_zeros' non-zero entries to zero.
    Assumes that each row has greater than 5 non-zero values

    Parameters:
    - matrix: A 2D NumPy array with users as rows and locations as columns.
    - max_zeros: The number of entries to set to zero per row.
    
    Returns:
    - A modified matrix with specified entries set to zero.
    """
    modified_matrix = np.copy(matrix)
    rows = modified_matrix.shape[0]

    for row_index in range(rows):
        nonzero_indices = np.where(modified_matrix[row_index][:] != 0 )[0]
        five_nonzero_indices = np.random.choice(nonzero_indices,n)
        #delete 5 from each row
        modified_matrix[row_index][five_nonzero_indices] = 0
        

    return modified_matrix

print("\nLoading Data\n")
#load original data
"""
df1 = pd.read_csv('../data/NYfiltered.csv')
print(df1)

print('Dataset 1 shape: {}'.format(df1.shape))
print('-Dataset examples-')
print(df1.iloc[::10000, :])
#print(df1['Date'].dtype)
df = df1



#Creating Data Matrix
df_matrix=pd.pivot_table(df,index='user',columns='location_id',aggfunc="size",fill_value=0)
print(df_matrix.shape)
"""

#load dataframe
with open('../data/pkl/NYfiltered.pkl', 'rb') as file:
    R = pickle.load(file)
print(R.shape) 
R1 = R



print("\nRandomly Delete 5 values in Set to create a test set\n")

R = zero_out(R1)



print("Original:\n",R1)
print("Test Set:\n",R)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(R, R1)
print("RMSE=",mse**0.5)

#train matrix factorization
print("\nTraining ...\n")
mf = MF(R, K=10, alpha=0.01, beta=0.01, iterations=1000)
training_process = mf.train()
L=np.rint(mf.full_matrix())
print("\nDone\n")

#analysis
x = [x for x, y in training_process]
y = [y for x, y in training_process]
x = x[::10]
y = y[::10]
plt.figure(figsize=((16,4)))
plt.plot(x, np.sqrt(y))
plt.xticks(x, x)
print("Minimizing Error on Training Set:\n")
plt.xlabel("Iterations")
plt.ylabel("Root Mean Square Error")
plt.grid(axis="y")
print("Learnt=\n",mf.full_matrix())
print("\nRating predictions=\n",L)
print()
print()

"""
# print("Global bias:")
# print(mf.b)
# print()
# print("User bias:")
# print(mf.b_u)
# print()
# print("Item bias:")
# print(mf.b_i)
print("\nFinding Error on test set...\n")
msef=0.0
# for i1 in range(len(i)):
#     for i2 in range(len(j)):
#         if R1.item(i[i1],j[i2])!=0:
#             msef = msef + (R1.item((i[i1],j[i2]))-(L).item((i[i1],j[i2])))**2
# msef = (msef/(len(j)*len(i)))
valid_cmp = ~np.isnan(df_matrix)
msef = np.sum(np.sum(np.multiply(valid_cmp,np.square(R1-L)),axis=None))/(len(j)*len(i)*1.00)

print("RMSE final=",msef**0.5)
"""




