import csv
import numpy as np
import pandas as pd
import pickle



csv_file_path = "../data/20plusfiltered.csv"
outputPath = csv_file_path[:-4]+'.pkl'

# Read CSV data into a list of lists
data = []
with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        data.append(row)


# Convert the data list into a NumPy array
matrix = np.array(data)
df = pd.DataFrame(data[1:],columns=data[0])

# Convert 'user' and 'location_id' columns to numeric type
df['user'] = pd.to_numeric(df['user'])
df['location_id'] = pd.to_numeric(df['location_id'])

# Create a pivot table to count occurrences of each user at each location
pivot_table = df.pivot_table(index='user', columns='location_id', aggfunc='size', fill_value=0)
print(pivot_table.describe())
# with open("../data/test_df.pkl", 'wb') as file:
#     pickle.dump(pivot_table, file)

# Convert the pivot table to a matrix
result_matrix = pivot_table.to_numpy()

# with open(outputPath, 'wb') as file:
    # pickle.dump(result_matrix, file)