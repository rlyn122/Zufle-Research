from ast import Module
import pandas as pd
import numpy as np
import csv
from fastai.collab import CollabDataLoaders
from fastai.tabular.all import *
import torch


file_path = ('../data/NYfiltered.csv')
print("\nLoading Data\n")

data = []
with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        data.append(row)

# Convert the data list into a NumPy array
matrix = np.array(data)

df = pd.DataFrame(data[1:],columns=data[0])

df['visits'] = df.groupby(['user','location_id'])['location_id'].transform('size')
df = df.drop_duplicates(['user','location_id']).reset_index(drop=True)
df['visits'] = pd.to_numeric(df["visits"])
print(df.head(10))

dls = CollabDataLoaders.from_df(df, user_name="user",item_name="location_id",rating_name="visits", bs=64)
dls.show_batch()

n_user = len(dls.classes["user"])
n_location_id = len(dls.classes["location_id"])
k = 10

user_factors = torch.rand(n_user, k)
location_id_factors = torch.rand(n_location_id, k)

class DotProduct(Module):
    def __init__(self, n_users, n_location_id, n_factors):
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_location_id, n_factors)
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        return (users * movies).sum(dim=1)
    
model = DotProduct(n_user, n_location_id, k)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3)