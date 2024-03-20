import numpy as np
import pandas as pd
import random

df_movie = pd.read_csv("./datasets/movielens/movies.csv")
print(df_movie.shape)
print(df_movie.head())

df_rating = pd.read_csv("./datasets/movielens/ratings.csv")
print(df_rating.shape)
print(df_rating.head())


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

df_movie = pd.read_csv("./datasets/movielens/movies.csv")
df_rating = pd.read_csv("./datasets/movielens/ratings.csv")

# Rest of the code from the notebook related to the recommender system