import torch
import torch.nn as nn
import json,os
import pandas as pd
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchtext import data
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Load JSON data from a file
businessdata = [] #add read size
with open(os.path.join('yelp/processed_business.json'), 'r') as f:
    for line in tqdm(f, desc="Loading business data"):
        businessdata.append(json.loads(line))

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(businessdata)

# Split the 'categories' field
df['categories'] = df['categories'].str.split(',')

# Create a binary matrix of businesses and categories
business_category_matrix = pd.get_dummies(df.set_index('business_id')['categories'].apply(pd.Series).stack()).groupby(level=0).sum()
# Compute the cosine similarity between each pair of businesses
similarity_matrix = cosine_similarity(business_category_matrix)

# Create a DataFrame from the similarity matrix
similarity_df = pd.DataFrame(similarity_matrix, index=business_category_matrix.index, columns=business_category_matrix.index)

print(similarity_df)
def recommend_businesses(business_id, num_recommendations=5):
    # Get the businesses most similar to the given business
    similar_businesses = similarity_df[business_id].sort_values(ascending=False)
    
    # Exclude the given business itself
    similar_businesses = similar_businesses.drop(business_id)
    
    # Return the top 'num_recommendations' businesses
    return similar_businesses.head(num_recommendations)
