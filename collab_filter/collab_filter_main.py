import sys
import pandas as pd
from scipy.spatial.distance import cosine

# Check if the file name is provided as an argument
if len(sys.argv) < 2:
    print("Usage: python collab_filter_main.py <csv_file>")
    sys.exit(1)

# Load the dataset
csv_file = sys.argv[1]
ratings = pd.read_csv(csv_file)

# Pivot the data to create a user-item matrix
user_item_matrix = ratings.pivot_table(index='asin', values=['avg_rating', 'activeness'])

# Fill missing values with 0
user_item_matrix = user_item_matrix.fillna(0)

# Function to calculate the cosine similarity between two users, considering activeness
def cosine_similarity(user1, user2, user1_activeness, user2_activeness):
    # Apply activeness as weights
    weighted_user1 = user1 * user1_activeness
    weighted_user2 = user2 * user2_activeness
    
    return 1 - cosine(weighted_user1, weighted_user2)

# Function to predict the rating for a user-item pair, considering activeness
def predict_rating(user_id, item_id, user_activeness):
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user_id]
    
    # Get the similarities between the user and all other users, considering activeness
    similarities = user_item_matrix.apply(lambda row: cosine_similarity(row, user_ratings, row['activeness'], user_activeness), axis=1)
    
    # Get the weighted average of the ratings for the item
    weighted_ratings = user_item_matrix[item_id] * similarities
    rating = weighted_ratings.sum() / similarities.sum()
    
    return rating

# Example usage
user_id = 1
item_id = 3
user_activeness = ratings.loc[ratings['user_id'] == user_id, 'activeness'].values[0]
predicted_rating = predict_rating(user_id, item_id, user_activeness)
print(f"Predicted rating for user {user_id} and item {item_id}: {predicted_rating}")
