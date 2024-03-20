import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Load the item details from the CSV file
item_details_df = pd.read_csv('processed_Gift_Cards.csv')

# Calculate item-item similarity matrix
item_features = item_details_df[['brand', 'avg_rating', 'activeness']].values

item_id = 'B00BXLVAD6'  # Replace with the desired item ID

# Function to recommend items based on collaborative filtering
def recommend_items(item_id, num_recommendations=5):
    # Find the index of the given item_id
    item_index = item_details_df[item_details_df['asin'] == item_id].index[0]

    # Calculate similarity scores for brand
    item_brand = item_features[item_index, [0]]
    brand_similarity = np.array([[1.0] if brand == item_brand else [0.0] for brand in item_features[:, [0]]])

    # Get the rating scores
    rating_scores = item_features[:, [1]]
    rating_scores_percent = rating_scores/5

    # Get the activeness scores
    activeness_scores = item_features[:, [2]]
    norm_activeness_scores = normalize(activeness_scores, axis=0)

    # Calculate weighted item-item similarity matrix
    item_rating = 0.2 * brand_similarity + 0.795 * rating_scores_percent + 0.05 * norm_activeness_scores
    item_rating[item_index] = 0

    # Sort the items based on similarity scores
    similar_item_indices = np.argsort(item_rating.flatten().tolist())[::-1]
    similar_item_sort = item_details_df.reindex(similar_item_indices)
    similar_item_select = similar_item_sort.head(num_recommendations)
    similar_item_select.index = range(1, num_recommendations+1)

    return similar_item_select

# Example usage
recommendations = recommend_items(item_id, num_recommendations=5)
print("Recommended Items:\n", recommendations)
