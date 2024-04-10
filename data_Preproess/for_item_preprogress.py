import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import torch
from tqdm import tqdm
from annoy import AnnoyIndex
import dask.dataframe as dd


# Function to flatten JSON
def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            for i, a in enumerate(x):
                if type(a) is dict:
                    flatten(a, name + str(i) + '_')
                else:
                    out[name + str(i)] = a
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

# Read the JSON file
flattened_data = []
with open('yelp/yelp_academic_dataset_business.json', 'r') as f:
    for i, line in enumerate(f):
        if i == 1000:  # stop after processing 1000 records
            break
        data = json.loads(line)
        flattened_data.append(flatten_json(data))

# Convert the flattened data to a DataFrame
df = pd.DataFrame(flattened_data)

# Fill missing values
df.fillna(0, inplace=True)
df['categories'] = df['categories'].str.split(', ')
# Convert the categories to separate columns
df_categories = df['categories'].apply(pd.Series)
df_categories = pd.get_dummies(df_categories.stack()).groupby(level=0).sum()

# Drop the original categories column and add the new ones
df = pd.concat([df.drop('categories', axis=1), df_categories], axis=1)

print(df.columns)

# Write the processed data to a new JSON file
df.to_json('processed3.json', orient='records', lines=True)

